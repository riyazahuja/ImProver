/-- If `A.det` has a constructive inverse, produce one for `A`. -/
def invertibleOfDetInvertible [Invertible A.det] : Invertible A where
  invOf := ⅟ A.det • A.adjugate
  mul_invOf_self := by
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝³ : Fintype n
      inst✝² : DecidableEq n
      inst✝¹ : CommRing α
      A B : Matrix n n α
      inst✝ : Invertible A.det
      ⊢ Eq (HMul.hMul A (HSMul.hSMul (Invertible.invOf A.det) A.adjugate)) 1
    -/
    rw [mul_smul_comm, mul_adjugate, smul_smul, invOf_mul_self, one_smul]
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝³ : Fintype n
      inst✝² : DecidableEq n
      inst✝¹ : CommRing α
      A B : Matrix n n α
      inst✝ : Invertible A.det
      ⊢ Eq (HMul.hMul (HSMul.hSMul (Invertible.invOf A.det) A.adjugate) A) 1
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  invOf_mul_self := by
    rw [smul_mul_assoc, adjugate_mul, smul_smul, invOf_mul_self, one_smul]


theorem invOf_eq [Invertible A.det] [Invertible A] : ⅟ A = ⅟ A.det • A.adjugate := by
  /-
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix n n α
    inst✝¹ : Invertible A.det
    inst✝ : Invertible A
    ⊢ Eq (Invertible.invOf A) (HSMul.hSMul (Invertible.invOf A.det) A.adjugate)
  -/
  letI := invertibleOfDetInvertible A
  /-
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix n n α
    inst✝¹ : Invertible A.det
    inst✝ : Invertible A
    this : Invertible A := A.invertibleOfDetInvertible
    ⊢ Eq (Invertible.invOf A) (HSMul.hSMul (Invertible.invOf A.det) A.adjugate)
  -/
  convert (rfl : ⅟ A = _)
  /-
    🎉 no goals
  -/


/-- `A.det` is invertible if `A` has a left inverse. -/
def detInvertibleOfLeftInverse (h : B * A = 1) : Invertible A.det where
  invOf := B.det
                       /-
                         l : Type u_1
                         m : Type u
                         n : Type u'
                         α : Type v
                         inst✝² : Fintype n
                         inst✝¹ : DecidableEq n
                         inst✝ : CommRing α
                         A B : Matrix n n α
                         h : Eq (HMul.hMul B A) 1
                         ⊢ Eq (HMul.hMul A.det B.det) 1
                       -/
                       /-
                         l : Type u_1
                         m : Type u
                         n : Type u'
                         α : Type v
                         inst✝² : Fintype n
                         inst✝¹ : DecidableEq n
                         inst✝ : CommRing α
                         A B : Matrix n n α
                         h : Eq (HMul.hMul B A) 1
                         ⊢ Eq (HMul.hMul B.det A.det) 1
                       -/
  mul_invOf_self := by rw [mul_comm, ← det_mul, h, det_one]
                       /-
                         🎉 no goals
                       -/
                       /-
                         🎉 no goals
                       -/
  invOf_mul_self := by rw [← det_mul, h, det_one]


/-- `A.det` is invertible if `A` has a right inverse. -/
def detInvertibleOfRightInverse (h : A * B = 1) : Invertible A.det where
  invOf := B.det
                       /-
                         l : Type u_1
                         m : Type u
                         n : Type u'
                         α : Type v
                         inst✝² : Fintype n
                         inst✝¹ : DecidableEq n
                         inst✝ : CommRing α
                         A B : Matrix n n α
                         h : Eq (HMul.hMul A B) 1
                         ⊢ Eq (HMul.hMul A.det B.det) 1
                       -/
                       /-
                         l : Type u_1
                         m : Type u
                         n : Type u'
                         α : Type v
                         inst✝² : Fintype n
                         inst✝¹ : DecidableEq n
                         inst✝ : CommRing α
                         A B : Matrix n n α
                         h : Eq (HMul.hMul A B) 1
                         ⊢ Eq (HMul.hMul B.det A.det) 1
                       -/
  mul_invOf_self := by rw [← det_mul, h, det_one]
                       /-
                         🎉 no goals
                       -/
                       /-
                         🎉 no goals
                       -/
  invOf_mul_self := by rw [mul_comm, ← det_mul, h, det_one]


/-- If `A` has a constructive inverse, produce one for `A.det`. -/
def detInvertibleOfInvertible [Invertible A] : Invertible A.det :=
  detInvertibleOfLeftInverse A (⅟ A) (invOf_mul_self _)


theorem det_invOf [Invertible A] [Invertible A.det] : (⅟ A).det = ⅟ A.det := by
  /-
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible A.det
    ⊢ Eq (Invertible.invOf A).det (Invertible.invOf A.det)
  -/
  letI := detInvertibleOfInvertible A
  /-
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    A : Matrix n n α
    inst✝¹ : Invertible A
    inst✝ : Invertible A.det
    this : Invertible A.det := A.detInvertibleOfInvertible
    ⊢ Eq (Invertible.invOf A).det (Invertible.invOf A.det)
  -/
  convert (rfl : _ = ⅟ A.det)
  /-
    🎉 no goals
  -/


/-- Together `Matrix.detInvertibleOfInvertible` and `Matrix.invertibleOfDetInvertible` form an
equivalence, although both sides of the equiv are subsingleton anyway. -/
@[simps]
def invertibleEquivDetInvertible : Invertible A ≃ Invertible A.det where
  toFun := @detInvertibleOfInvertible _ _ _ _ _ A
  invFun := @invertibleOfDetInvertible _ _ _ _ _ A
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- Given a proof that `A.det` has a constructive inverse, lift `A` to `(Matrix n n α)ˣ`-/
def unitOfDetInvertible [Invertible A.det] : (Matrix n n α)ˣ :=
  @unitOfInvertible _ _ A (invertibleOfDetInvertible A)


/-- When lowered to a prop, `Matrix.invertibleEquivDetInvertible` forms an `iff`. -/
theorem isUnit_iff_isUnit_det : IsUnit A ↔ IsUnit A.det := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Iff (IsUnit A) (IsUnit A.det)
  -/
  simp only [← nonempty_invertible_iff_isUnit, (invertibleEquivDetInvertible A).nonempty_congr]
  /-
    🎉 no goals
  -/


@[simp]
theorem isUnits_det_units (A : (Matrix n n α)ˣ) : IsUnit (A : Matrix n n α).det :=
  isUnit_iff_isUnit_det _ |>.mp A.isUnit


theorem isUnit_det_of_invertible [Invertible A] : IsUnit A.det :=
  @isUnit_of_invertible _ _ _ (detInvertibleOfInvertible A)


theorem isUnit_det_of_left_inverse (h : B * A = 1) : IsUnit A.det :=
  @isUnit_of_invertible _ _ _ (detInvertibleOfLeftInverse _ _ h)


theorem isUnit_det_of_right_inverse (h : A * B = 1) : IsUnit A.det :=
  @isUnit_of_invertible _ _ _ (detInvertibleOfRightInverse _ _ h)


theorem det_ne_zero_of_left_inverse [Nontrivial α] (h : B * A = 1) : A.det ≠ 0 :=
  (isUnit_det_of_left_inverse h).ne_zero


theorem det_ne_zero_of_right_inverse [Nontrivial α] (h : A * B = 1) : A.det ≠ 0 :=
  (isUnit_det_of_right_inverse h).ne_zero


theorem isUnit_det_transpose (h : IsUnit A.det) : IsUnit Aᵀ.det := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ IsUnit A.transpose.det
  -/
  rw [det_transpose]
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ IsUnit A.det
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- The inverse of a square matrix, when it is invertible (and zero otherwise). -/
noncomputable instance inv : Inv (Matrix n n α) :=
  ⟨fun A => Ring.inverse A.det • A.adjugate⟩


theorem inv_def (A : Matrix n n α) : A⁻¹ = Ring.inverse A.det • A.adjugate :=
  rfl


theorem nonsing_inv_apply_not_isUnit (h : ¬IsUnit A.det) : A⁻¹ = 0 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Not (IsUnit A.det)
    ⊢ Eq (Inv.inv A) 0
  -/
  rw [inv_def, Ring.inverse_non_unit _ h, zero_smul]
  /-
    🎉 no goals
  -/


theorem nonsing_inv_apply (h : IsUnit A.det) : A⁻¹ = (↑h.unit⁻¹ : α) • A.adjugate := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (Inv.inv A) (HSMul.hSMul (↑(Inv.inv h.unit)) A.adjugate)
  -/
  rw [inv_def, ← Ring.inverse_unit h.unit, IsUnit.unit_spec]
  /-
    🎉 no goals
  -/


/-- The nonsingular inverse is the same as `invOf` when `A` is invertible. -/
@[simp]
theorem invOf_eq_nonsing_inv [Invertible A] : ⅟ A = A⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A
    ⊢ Eq (Invertible.invOf A) (Inv.inv A)
  -/
  letI := detInvertibleOfInvertible A
  /-
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A
    this : Invertible A.det := A.detInvertibleOfInvertible
    ⊢ Eq (Invertible.invOf A) (Inv.inv A)
  -/
  rw [inv_def, Ring.inverse_invertible, invOf_eq]
  /-
    🎉 no goals
  -/


/-- Coercing the result of `Units.instInv` is the same as coercing first and applying the
nonsingular inverse. -/
@[simp, norm_cast]
theorem coe_units_inv (A : (Matrix n n α)ˣ) : ↑A⁻¹ = (A⁻¹ : Matrix n n α) := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Units (Matrix n n α)
    ⊢ Eq (↑(Inv.inv A)) (Inv.inv ↑A)
  -/
  letI := A.invertible
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Units (Matrix n n α)
    this : Invertible ↑A := A.invertible
    ⊢ Eq (↑(Inv.inv A)) (Inv.inv ↑A)
  -/
  rw [← invOf_eq_nonsing_inv, invOf_units]
  /-
    🎉 no goals
  -/


/-- The nonsingular inverse is the same as the general `Ring.inverse`. -/
theorem nonsing_inv_eq_ring_inverse : A⁻¹ = Ring.inverse A := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (Inv.inv A) (Ring.inverse A)
  -/
  by_cases h_det : IsUnit A.det
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h_det : IsUnit A.det
      ⊢ Eq (Inv.inv A) (Ring.inverse A)
    -/
  · cases (A.isUnit_iff_isUnit_det.mpr h_det).nonempty_invertible
    /-
      case pos.intro
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h_det : IsUnit A.det
      val✝ : Invertible A
      ⊢ Eq (Inv.inv A) (Ring.inverse A)
    -/
    rw [← invOf_eq_nonsing_inv, Ring.inverse_invertible]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h_det : Not (IsUnit A.det)
      ⊢ Eq (Inv.inv A) (Ring.inverse A)
    -/
  · have h := mt A.isUnit_iff_isUnit_det.mp h_det
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h_det : Not (IsUnit A.det)
      h : Not (IsUnit A)
      ⊢ Eq (Inv.inv A) (Ring.inverse A)
    -/
    rw [Ring.inverse_non_unit _ h, nonsing_inv_apply_not_isUnit A h_det]
    /-
      🎉 no goals
    -/


theorem transpose_nonsing_inv : A⁻¹ᵀ = Aᵀ⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (Inv.inv A).transpose (Inv.inv A.transpose)
  -/
  rw [inv_def, inv_def, transpose_smul, det_transpose, adjugate_transpose]
  /-
    🎉 no goals
  -/


theorem conjTranspose_nonsing_inv [StarRing α] : A⁻¹ᴴ = Aᴴ⁻¹ := by
  rw [inv_def, inv_def, conjTranspose_smul, det_conjTranspose, adjugate_conjTranspose,
    Ring.inverse_star]


/-- The `nonsing_inv` of `A` is a right inverse. -/
@[simp]
theorem mul_nonsing_inv (h : IsUnit A.det) : A * A⁻¹ = 1 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul A (Inv.inv A)) 1
  -/
  cases (A.isUnit_iff_isUnit_det.mpr h).nonempty_invertible
  /-
    case intro
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    val✝ : Invertible A
    ⊢ Eq (HMul.hMul A (Inv.inv A)) 1
  -/
  rw [← invOf_eq_nonsing_inv, mul_invOf_self]
  /-
    🎉 no goals
  -/


/-- The `nonsing_inv` of `A` is a left inverse. -/
@[simp]
theorem nonsing_inv_mul (h : IsUnit A.det) : A⁻¹ * A = 1 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (Inv.inv A) A) 1
  -/
  cases (A.isUnit_iff_isUnit_det.mpr h).nonempty_invertible
  /-
    case intro
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    val✝ : Invertible A
    ⊢ Eq (HMul.hMul (Inv.inv A) A) 1
  -/
  rw [← invOf_eq_nonsing_inv, invOf_mul_self]
  /-
    🎉 no goals
  -/


instance [Invertible A] : Invertible A⁻¹ := by
  /-
    l : Type u_1
    m : Type u
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A B : Matrix n n α
    inst✝ : Invertible A
    ⊢ Invertible (Inv.inv A)
  -/
  rw [← invOf_eq_nonsing_inv]
  /-
    l : Type u_1
    m : Type u
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A B : Matrix n n α
    inst✝ : Invertible A
    ⊢ Invertible (Invertible.invOf A)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_inv_of_invertible [Invertible A] : A⁻¹⁻¹ = A := by
  /-
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A
    ⊢ Eq (Inv.inv (Inv.inv A)) A
  -/
  simp only [← invOf_eq_nonsing_inv, invOf_invOf]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_nonsing_inv_cancel_right (B : Matrix m n α) (h : IsUnit A.det) : B * A * A⁻¹ = B := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    B : Matrix m n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (HMul.hMul B A) (Inv.inv A)) B
  -/
  simp [Matrix.mul_assoc, mul_nonsing_inv A h]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_nonsing_inv_cancel_left (B : Matrix n m α) (h : IsUnit A.det) : A * (A⁻¹ * B) = B := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    B : Matrix n m α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul A (HMul.hMul (Inv.inv A) B)) B
  -/
  simp [← Matrix.mul_assoc, mul_nonsing_inv A h]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonsing_inv_mul_cancel_right (B : Matrix m n α) (h : IsUnit A.det) : B * A⁻¹ * A = B := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    B : Matrix m n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (HMul.hMul B (Inv.inv A)) A) B
  -/
  simp [Matrix.mul_assoc, nonsing_inv_mul A h]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonsing_inv_mul_cancel_left (B : Matrix n m α) (h : IsUnit A.det) : A⁻¹ * (A * B) = B := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    B : Matrix n m α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (Inv.inv A) (HMul.hMul A B)) B
  -/
  simp [← Matrix.mul_assoc, nonsing_inv_mul A h]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_inv_of_invertible [Invertible A] : A * A⁻¹ = 1 :=
  mul_nonsing_inv A (isUnit_det_of_invertible A)


@[simp]
theorem inv_mul_of_invertible [Invertible A] : A⁻¹ * A = 1 :=
  nonsing_inv_mul A (isUnit_det_of_invertible A)


@[simp]
theorem mul_inv_cancel_right_of_invertible (B : Matrix m n α) [Invertible A] : B * A * A⁻¹ = B :=
  mul_nonsing_inv_cancel_right A B (isUnit_det_of_invertible A)


@[simp]
theorem mul_inv_cancel_left_of_invertible (B : Matrix n m α) [Invertible A] : A * (A⁻¹ * B) = B :=
  mul_nonsing_inv_cancel_left A B (isUnit_det_of_invertible A)


@[simp]
theorem inv_mul_cancel_right_of_invertible (B : Matrix m n α) [Invertible A] : B * A⁻¹ * A = B :=
  nonsing_inv_mul_cancel_right A B (isUnit_det_of_invertible A)


@[simp]
theorem inv_mul_cancel_left_of_invertible (B : Matrix n m α) [Invertible A] : A⁻¹ * (A * B) = B :=
  nonsing_inv_mul_cancel_left A B (isUnit_det_of_invertible A)


theorem inv_mul_eq_iff_eq_mul_of_invertible (A B C : Matrix n n α) [Invertible A] :
    A⁻¹ * B = C ↔ B = A * C :=
               /-
                 n : Type u'
                 α : Type v
                 inst✝³ : Fintype n
                 inst✝² : DecidableEq n
                 inst✝¹ : CommRing α
                 A B C : Matrix n n α
                 inst✝ : Invertible A
                 h : Eq (HMul.hMul (Inv.inv A) B) C
                 ⊢ Eq B (HMul.hMul A C)
               -/
  ⟨fun h => by rw [← h, mul_inv_cancel_left_of_invertible],
               /-
                 🎉 no goals
               -/
               /-
                 n : Type u'
                 α : Type v
                 inst✝³ : Fintype n
                 inst✝² : DecidableEq n
                 inst✝¹ : CommRing α
                 A B C : Matrix n n α
                 inst✝ : Invertible A
                 h : Eq B (HMul.hMul A C)
                 ⊢ Eq (HMul.hMul (Inv.inv A) B) C
               -/
   fun h => by rw [h, inv_mul_cancel_left_of_invertible]⟩
               /-
                 🎉 no goals
               -/


theorem mul_inv_eq_iff_eq_mul_of_invertible (A B C : Matrix n n α) [Invertible A] :
    B * A⁻¹ = C ↔ B = C * A :=
               /-
                 n : Type u'
                 α : Type v
                 inst✝³ : Fintype n
                 inst✝² : DecidableEq n
                 inst✝¹ : CommRing α
                 A B C : Matrix n n α
                 inst✝ : Invertible A
                 h : Eq (HMul.hMul B (Inv.inv A)) C
                 ⊢ Eq B (HMul.hMul C A)
               -/
  ⟨fun h => by rw [← h, inv_mul_cancel_right_of_invertible],
               /-
                 🎉 no goals
               -/
               /-
                 n : Type u'
                 α : Type v
                 inst✝³ : Fintype n
                 inst✝² : DecidableEq n
                 inst✝¹ : CommRing α
                 A B C : Matrix n n α
                 inst✝ : Invertible A
                 h : Eq B (HMul.hMul C A)
                 ⊢ Eq (HMul.hMul B (Inv.inv A)) C
               -/
   fun h => by rw [h, mul_inv_cancel_right_of_invertible]⟩
               /-
                 🎉 no goals
               -/


lemma inv_mulVec_eq_vec {A : Matrix n n α} [Invertible A]
    {u v : n → α} (hM : u = A.mulVec v) : A⁻¹.mulVec u = v := by
  /-
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A
    u v : n → α
    hM : Eq u (A.mulVec v)
    ⊢ Eq ((Inv.inv A).mulVec u) v
  -/
  rw [hM, Matrix.mulVec_mulVec, Matrix.inv_mul_of_invertible, Matrix.one_mulVec]
  /-
    🎉 no goals
  -/


lemma mul_right_injective_of_invertible [Invertible A] :
    Function.Injective (fun (x : Matrix n m α) => A * x) :=
                  /-
                    m : Type u
                    n : Type u'
                    α : Type v
                    inst✝³ : Fintype n
                    inst✝² : DecidableEq n
                    inst✝¹ : CommRing α
                    A : Matrix n n α
                    inst✝ : Invertible A
                    x✝¹ x✝ : Matrix n m α
                    h : Eq ((fun x => HMul.hMul A x) x✝¹) ((fun x => HMul.hMul A x) x✝)
                    ⊢ Eq x✝¹ x✝
                  -/
  fun _ _ h => by simpa only [inv_mul_cancel_left_of_invertible] using congr_arg (A⁻¹ * ·) h
                  /-
                    🎉 no goals
                  -/


lemma mul_left_injective_of_invertible [Invertible A] :
    Function.Injective (fun (x : Matrix m n α) => x * A) :=
                    /-
                      m : Type u
                      n : Type u'
                      α : Type v
                      inst✝³ : Fintype n
                      inst✝² : DecidableEq n
                      inst✝¹ : CommRing α
                      A : Matrix n n α
                      inst✝ : Invertible A
                      a x : Matrix m n α
                      hax : Eq ((fun x => HMul.hMul x A) a) ((fun x => HMul.hMul x A) x)
                      ⊢ Eq a x
                    -/
  fun a x hax => by simpa only [mul_inv_cancel_right_of_invertible] using congr_arg (· * A⁻¹) hax
                    /-
                      🎉 no goals
                    -/


lemma mul_right_inj_of_invertible [Invertible A] {x y : Matrix n m α} : A * x = A * y ↔ x = y :=
  (mul_right_injective_of_invertible A).eq_iff


lemma mul_left_inj_of_invertible [Invertible A] {x y : Matrix m n α} : x * A = y * A ↔ x = y :=
  (mul_left_injective_of_invertible A).eq_iff


lemma mul_left_injective_of_inv (A : Matrix m n α) (B : Matrix n m α) (h : A * B = 1) :
    Function.Injective (fun x : Matrix l m α => x * A) := fun _ _ g => by
  /-
    l : Type u_1
    m : Type u
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : CommRing α
    A : Matrix m n α
    B : Matrix n m α
    h : Eq (HMul.hMul A B) 1
    x✝¹ x✝ : Matrix l m α
    g : Eq ((fun x => HMul.hMul x A) x✝¹) ((fun x => HMul.hMul x A) x✝)
    ⊢ Eq x✝¹ x✝
  -/
  simpa only [Matrix.mul_assoc, Matrix.mul_one, h] using congr_arg (· * B) g
  /-
    🎉 no goals
  -/


lemma mul_right_injective_of_inv (A : Matrix m n α) (B : Matrix n m α) (h : A * B = 1) :
    Function.Injective (fun x : Matrix m l α => B * x) :=
                  /-
                    l : Type u_1
                    m : Type u
                    n : Type u'
                    α : Type v
                    inst✝³ : Fintype n
                    inst✝² : Fintype m
                    inst✝¹ : DecidableEq m
                    inst✝ : CommRing α
                    A : Matrix m n α
                    B : Matrix n m α
                    h : Eq (HMul.hMul A B) 1
                    x✝¹ x✝ : Matrix m l α
                    g : Eq ((fun x => HMul.hMul B x) x✝¹) ((fun x => HMul.hMul B x) x✝)
                    ⊢ Eq x✝¹ x✝
                  -/
  fun _ _ g => by simpa only [← Matrix.mul_assoc, Matrix.one_mul, h] using congr_arg (A * ·) g
                  /-
                    🎉 no goals
                  -/


theorem vecMul_surjective_iff_exists_left_inverse
    [DecidableEq n] [Fintype m] [Finite n] {A : Matrix m n R} :
    Function.Surjective A.vecMul ↔ ∃ B : Matrix n m R, B * A = 1 := by
  /-
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    ⊢ Iff (Function.Surjective fun v => Matrix.vecMul v A) (Exists fun B => Eq (HM …
  -/
  cases nonempty_fintype n
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    val✝ : Fintype n
    ⊢ Iff (Function.Surjective fun v => Matrix.vecMul v A) (Exists fun B => Eq (HM …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨B, hBA⟩ y ↦ ⟨y ᵥ* B, by simp [hBA]⟩⟩
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    val✝ : Fintype n
    h : Function.Surjective fun v => Matrix.vecMul v A
    ⊢ Exists fun B => Eq (HMul.hMul B A) 1
  -/
  choose rows hrows using (h <| Pi.single · 1)
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    val✝ : Fintype n
    h : Function.Surjective fun v => Matrix.vecMul v A
    rows : n → m → R
    hrows : ∀ (x : n), Eq ((fun v => Matrix.vecMul v A) (rows x)) (Pi.single x 1)
    ⊢ Exists fun B => Eq (HMul.hMul B A) 1
  -/
  refine ⟨Matrix.of rows, Matrix.ext fun i j => ?_⟩
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    val✝ : Fintype n
    h : Function.Surjective fun v => Matrix.vecMul v A
    rows : n → m → R
    hrows : ∀ (x : n), Eq ((fun v => Matrix.vecMul v A) (rows x)) (Pi.single x 1)
    i j : n
    ⊢ Eq (HMul.hMul (Matrix.of rows) A i j) (1 i j)
  -/
  rw [mul_apply_eq_vecMul, one_eq_pi_single, ← hrows]
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq n
    inst✝¹ : Fintype m
    inst✝ : Finite n
    A : Matrix m n R
    val✝ : Fintype n
    h : Function.Surjective fun v => Matrix.vecMul v A
    rows : n → m → R
    hrows : ∀ (x : n), Eq ((fun v => Matrix.vecMul v A) (rows x)) (Pi.single x 1)
    i j : n
    ⊢ Eq (Matrix.vecMul (Matrix.of rows i) A j) ((fun v => Matrix.vecMul v A) (row …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mulVec_surjective_iff_exists_right_inverse
    [DecidableEq m] [Finite m] [Fintype n] {A : Matrix m n R} :
    Function.Surjective A.mulVec ↔ ∃ B : Matrix n m R, A * B = 1 := by
  /-
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    ⊢ Iff (Function.Surjective A.mulVec) (Exists fun B => Eq (HMul.hMul A B) 1)
  -/
  cases nonempty_fintype m
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    val✝ : Fintype m
    ⊢ Iff (Function.Surjective A.mulVec) (Exists fun B => Eq (HMul.hMul A B) 1)
  -/
  refine ⟨fun h ↦ ?_, fun ⟨B, hBA⟩ y ↦ ⟨B *ᵥ y, by simp [hBA]⟩⟩
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    val✝ : Fintype m
    h : Function.Surjective A.mulVec
    ⊢ Exists fun B => Eq (HMul.hMul A B) 1
  -/
  choose cols hcols using (h <| Pi.single · 1)
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    val✝ : Fintype m
    h : Function.Surjective A.mulVec
    cols : m → n → R
    hcols : ∀ (x : m), Eq (A.mulVec (cols x)) (Pi.single x 1)
    ⊢ Exists fun B => Eq (HMul.hMul A B) 1
  -/
  refine ⟨(Matrix.of cols)ᵀ, Matrix.ext fun i j ↦ ?_⟩
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    val✝ : Fintype m
    h : Function.Surjective A.mulVec
    cols : m → n → R
    hcols : ∀ (x : m), Eq (A.mulVec (cols x)) (Pi.single x 1)
    i j : m
    ⊢ Eq (HMul.hMul A (Matrix.of cols).transpose i j) (1 i j)
  -/
  rw [one_eq_pi_single, Pi.single_comm, ← hcols j]
  /-
    case intro
    m : Type u
    n : Type u'
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : DecidableEq m
    inst✝¹ : Finite m
    inst✝ : Fintype n
    A : Matrix m n R
    val✝ : Fintype m
    h : Function.Surjective A.mulVec
    cols : m → n → R
    hcols : ∀ (x : m), Eq (A.mulVec (cols x)) (Pi.single x 1)
    i j : m
    ⊢ Eq (HMul.hMul A (Matrix.of cols).transpose i j) (A.mulVec (cols j) i)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem vecMul_surjective_iff_isUnit {A : Matrix m m R} :
    Function.Surjective A.vecMul ↔ IsUnit A := by
  /-
    m : Type u
    inst✝² : DecidableEq m
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype m
    A : Matrix m m R
    ⊢ Iff (Function.Surjective fun v => Matrix.vecMul v A) (IsUnit A)
  -/
  rw [vecMul_surjective_iff_exists_left_inverse, exists_left_inverse_iff_isUnit]
  /-
    🎉 no goals
  -/


theorem mulVec_surjective_iff_isUnit {A : Matrix m m R} :
    Function.Surjective A.mulVec ↔ IsUnit A := by
  /-
    m : Type u
    inst✝² : DecidableEq m
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype m
    A : Matrix m m R
    ⊢ Iff (Function.Surjective A.mulVec) (IsUnit A)
  -/
  rw [mulVec_surjective_iff_exists_right_inverse, exists_right_inverse_iff_isUnit]
  /-
    🎉 no goals
  -/


theorem vecMul_injective_iff_isUnit {A : Matrix m m K} :
    Function.Injective A.vecMul ↔ IsUnit A := by
  /-
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    ⊢ Iff (Function.Injective fun v => Matrix.vecMul v A) (IsUnit A)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      m : Type u
      inst✝² : DecidableEq m
      K : Type u_3
      inst✝¹ : Field K
      inst✝ : Fintype m
      A : Matrix m m K
      h : Function.Injective fun v => Matrix.vecMul v A
      ⊢ IsUnit A
    -/
  · rw [← vecMul_surjective_iff_isUnit]
    /-
      case refine_1
      m : Type u
      inst✝² : DecidableEq m
      K : Type u_3
      inst✝¹ : Field K
      inst✝ : Fintype m
      A : Matrix m m K
      h : Function.Injective fun v => Matrix.vecMul v A
      ⊢ Function.Surjective fun v => Matrix.vecMul v A
    -/
    exact LinearMap.surjective_of_injective (f := A.vecMulLinear) h
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    h : IsUnit A
    ⊢ Function.Injective fun v => Matrix.vecMul v A
  -/
  change Function.Injective A.vecMulLinear
  /-
    case refine_2
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    h : IsUnit A
    ⊢ Function.Injective ⇑A.vecMulLinear
  -/
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  /-
    case refine_2
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    h : IsUnit A
    ⊢ ∀ (m_1 : m → K), Eq (A.vecMulLinear m_1) 0 → Eq m_1 0
  -/
  intro c hc
  /-
    case refine_2
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    h : IsUnit A
    c : m → K
    hc : Eq (A.vecMulLinear c) 0
    ⊢ Eq c 0
  -/
  replace h := h.invertible
  /-
    case refine_2
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    c : m → K
    hc : Eq (A.vecMulLinear c) 0
    h : Invertible A
    ⊢ Eq c 0
  -/
  simpa using congr_arg A⁻¹.vecMulLinear hc
  /-
    🎉 no goals
  -/


theorem mulVec_injective_iff_isUnit {A : Matrix m m K} :
    Function.Injective A.mulVec ↔ IsUnit A := by
  /-
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    ⊢ Iff (Function.Injective A.mulVec) (IsUnit A)
  -/
  rw [← isUnit_transpose, ← vecMul_injective_iff_isUnit]
  /-
    m : Type u
    inst✝² : DecidableEq m
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Fintype m
    A : Matrix m m K
    ⊢ Iff (Function.Injective A.mulVec) (Function.Injective fun v => Matrix.vecMul …
  -/
  simp_rw [vecMul_transpose]
  /-
    🎉 no goals
  -/


theorem linearIndependent_rows_iff_isUnit {A : Matrix m m K} :
    LinearIndependent K (fun i ↦ A i) ↔ IsUnit A := by
  rw [← transpose_transpose A, ← mulVec_injective_iff, ← coe_mulVecLin, mulVecLin_transpose,
    transpose_transpose, ← vecMul_injective_iff_isUnit, coe_vecMulLinear]


theorem linearIndependent_cols_iff_isUnit {A : Matrix m m K} :
    LinearIndependent K (fun i ↦ Aᵀ i) ↔ IsUnit A := by
  rw [← transpose_transpose A, isUnit_transpose, linearIndependent_rows_iff_isUnit,
    transpose_transpose]


theorem vecMul_surjective_of_invertible (A : Matrix m m R) [Invertible A] :
    Function.Surjective A.vecMul :=
  vecMul_surjective_iff_isUnit.2 <| isUnit_of_invertible A


theorem mulVec_surjective_of_invertible (A : Matrix m m R) [Invertible A] :
    Function.Surjective A.mulVec :=
  mulVec_surjective_iff_isUnit.2 <| isUnit_of_invertible A


theorem vecMul_injective_of_invertible (A : Matrix m m K) [Invertible A] :
    Function.Injective A.vecMul :=
  vecMul_injective_iff_isUnit.2 <| isUnit_of_invertible A


theorem mulVec_injective_of_invertible (A : Matrix m m K) [Invertible A] :
    Function.Injective A.mulVec :=
  mulVec_injective_iff_isUnit.2 <| isUnit_of_invertible A


theorem linearIndependent_rows_of_invertible (A : Matrix m m K) [Invertible A] :
    LinearIndependent K (fun i ↦ A i) :=
  linearIndependent_rows_iff_isUnit.2 <| isUnit_of_invertible A


theorem linearIndependent_cols_of_invertible (A : Matrix m m K) [Invertible A] :
    LinearIndependent K (fun i ↦ Aᵀ i) :=
  linearIndependent_cols_iff_isUnit.2 <| isUnit_of_invertible A


theorem nonsing_inv_cancel_or_zero : A⁻¹ * A = 1 ∧ A * A⁻¹ = 1 ∨ A⁻¹ = 0 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Or (And (Eq (HMul.hMul (Inv.inv A) A) 1) (Eq (HMul.hMul A (Inv.inv A)) 1)) ( …
  -/
  by_cases h : IsUnit A.det
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : IsUnit A.det
      ⊢ Or (And (Eq (HMul.hMul (Inv.inv A) A) 1) (Eq (HMul.hMul A (Inv.inv A)) 1)) ( …
    -/
  · exact Or.inl ⟨nonsing_inv_mul _ h, mul_nonsing_inv _ h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Not (IsUnit A.det)
      ⊢ Or (And (Eq (HMul.hMul (Inv.inv A) A) 1) (Eq (HMul.hMul A (Inv.inv A)) 1)) ( …
    -/
  · exact Or.inr (nonsing_inv_apply_not_isUnit _ h)
    /-
      🎉 no goals
    -/


theorem det_nonsing_inv_mul_det (h : IsUnit A.det) : A⁻¹.det * A.det = 1 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (Inv.inv A).det A.det) 1
  -/
  rw [← det_mul, A.nonsing_inv_mul h, det_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem det_nonsing_inv : A⁻¹.det = Ring.inverse A.det := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
  -/
  by_cases h : IsUnit A.det
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : IsUnit A.det
      ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
    -/
  · cases h.nonempty_invertible
    /-
      case pos.intro
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : IsUnit A.det
      val✝ : Invertible A.det
      ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
    -/
    letI := invertibleOfDetInvertible A
    /-
      case pos.intro
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : IsUnit A.det
      val✝ : Invertible A.det
      this : Invertible A := A.invertibleOfDetInvertible
      ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
    -/
    rw [Ring.inverse_invertible, ← invOf_eq_nonsing_inv, det_invOf]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : Not (IsUnit A.det)
    ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
  -/
  cases isEmpty_or_nonempty n
    /-
      case neg.inl
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Not (IsUnit A.det)
      h✝ : IsEmpty n
      ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
    -/
  · rw [det_isEmpty, det_isEmpty, Ring.inverse_one]
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Not (IsUnit A.det)
      h✝ : Nonempty n
      ⊢ Eq (Inv.inv A).det (Ring.inverse A.det)
    -/
  · rw [Ring.inverse_non_unit _ h, nonsing_inv_apply_not_isUnit _ h, det_zero ‹_›]
    /-
      🎉 no goals
    -/


theorem isUnit_nonsing_inv_det (h : IsUnit A.det) : IsUnit A⁻¹.det :=
  isUnit_of_mul_eq_one _ _ (A.det_nonsing_inv_mul_det h)


@[simp]
theorem nonsing_inv_nonsing_inv (h : IsUnit A.det) : A⁻¹⁻¹ = A :=
  calc
                            /-
                              n : Type u'
                              α : Type v
                              inst✝² : Fintype n
                              inst✝¹ : DecidableEq n
                              inst✝ : CommRing α
                              A : Matrix n n α
                              h : IsUnit A.det
                              ⊢ Eq (Inv.inv (Inv.inv A)) (HMul.hMul 1 (Inv.inv (Inv.inv A)))
                            -/
    A⁻¹⁻¹ = 1 * A⁻¹⁻¹ := by rw [Matrix.one_mul]
                            /-
                              🎉 no goals
                            -/
                              /-
                                n : Type u'
                                α : Type v
                                inst✝² : Fintype n
                                inst✝¹ : DecidableEq n
                                inst✝ : CommRing α
                                A : Matrix n n α
                                h : IsUnit A.det
                                ⊢ Eq (HMul.hMul 1 (Inv.inv (Inv.inv A))) (HMul.hMul (HMul.hMul A (Inv.inv A))  …
                              -/
    _ = A * A⁻¹ * A⁻¹⁻¹ := by rw [A.mul_nonsing_inv h]
                              /-
                                🎉 no goals
                              -/
    _ = A := by
      /-
        n : Type u'
        α : Type v
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        inst✝ : CommRing α
        A : Matrix n n α
        h : IsUnit A.det
        ⊢ Eq (HMul.hMul (HMul.hMul A (Inv.inv A)) (Inv.inv (Inv.inv A))) A
      -/
      rw [Matrix.mul_assoc, A⁻¹.mul_nonsing_inv (A.isUnit_nonsing_inv_det h), Matrix.mul_one]
      /-
        🎉 no goals
      -/


theorem isUnit_nonsing_inv_det_iff {A : Matrix n n α} : IsUnit A⁻¹.det ↔ IsUnit A.det := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Iff (IsUnit (Inv.inv A).det) (IsUnit A.det)
  -/
  rw [Matrix.det_nonsing_inv, isUnit_ring_inverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem isUnit_nonsing_inv_iff {A : Matrix n n α} : IsUnit A⁻¹ ↔ IsUnit A := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Iff (IsUnit (Inv.inv A)) (IsUnit A)
  -/
  simp_rw [isUnit_iff_isUnit_det, isUnit_nonsing_inv_det_iff]
  /-
    🎉 no goals
  -/

-- `IsUnit.invertible` lifts the proposition `IsUnit A` to a constructive inverse of `A`.

/-- A version of `Matrix.invertibleOfDetInvertible` with the inverse defeq to `A⁻¹` that is
therefore noncomputable. -/
noncomputable def invertibleOfIsUnitDet (h : IsUnit A.det) : Invertible A :=
  ⟨A⁻¹, nonsing_inv_mul A h, mul_nonsing_inv A h⟩


/-- A version of `Matrix.unitOfDetInvertible` with the inverse defeq to `A⁻¹` that is therefore
noncomputable. -/
noncomputable def nonsingInvUnit (h : IsUnit A.det) : (Matrix n n α)ˣ :=
  @unitOfInvertible _ _ _ (invertibleOfIsUnitDet A h)


theorem unitOfDetInvertible_eq_nonsingInvUnit [Invertible A.det] :
    unitOfDetInvertible A = nonsingInvUnit A (isUnit_of_invertible _) := by
  /-
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A.det
    ⊢ Eq A.unitOfDetInvertible (A.nonsingInvUnit ⋯)
  -/
  ext
  /-
    case a.a
    n : Type u'
    α : Type v
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing α
    A : Matrix n n α
    inst✝ : Invertible A.det
    i✝ j✝ : n
    ⊢ Eq (↑A.unitOfDetInvertible i✝ j✝) (↑(A.nonsingInvUnit ⋯) i✝ j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If matrix A is left invertible, then its inverse equals its left inverse. -/
theorem inv_eq_left_inv (h : B * A = 1) : A⁻¹ = B :=
  letI := invertibleOfLeftInverse _ _ h
  invOf_eq_nonsing_inv A ▸ invOf_eq_left_inv h


/-- If matrix A is right invertible, then its inverse equals its right inverse. -/
theorem inv_eq_right_inv (h : A * B = 1) : A⁻¹ = B :=
  inv_eq_left_inv (mul_eq_one_comm.2 h)


/-- The left inverse of matrix A is unique when existing. -/
theorem left_inv_eq_left_inv (h : B * A = 1) (g : C * A = 1) : B = C := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B C : Matrix n n α
    h : Eq (HMul.hMul B A) 1
    g : Eq (HMul.hMul C A) 1
    ⊢ Eq B C
  -/
  rw [← inv_eq_left_inv h, ← inv_eq_left_inv g]
  /-
    🎉 no goals
  -/


/-- The right inverse of matrix A is unique when existing. -/
theorem right_inv_eq_right_inv (h : A * B = 1) (g : A * C = 1) : B = C := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B C : Matrix n n α
    h : Eq (HMul.hMul A B) 1
    g : Eq (HMul.hMul A C) 1
    ⊢ Eq B C
  -/
  rw [← inv_eq_right_inv h, ← inv_eq_right_inv g]
  /-
    🎉 no goals
  -/


/-- The right inverse of matrix A equals the left inverse of A when they exist. -/
theorem right_inv_eq_left_inv (h : A * B = 1) (g : C * A = 1) : B = C := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B C : Matrix n n α
    h : Eq (HMul.hMul A B) 1
    g : Eq (HMul.hMul C A) 1
    ⊢ Eq B C
  -/
  rw [← inv_eq_right_inv h, ← inv_eq_left_inv g]
  /-
    🎉 no goals
  -/


theorem inv_inj (h : A⁻¹ = B⁻¹) (h' : IsUnit A.det) : A = B := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Eq (Inv.inv A) (Inv.inv B)
    h' : IsUnit A.det
    ⊢ Eq A B
  -/
  refine left_inv_eq_left_inv (mul_nonsing_inv _ h') ?_
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Eq (Inv.inv A) (Inv.inv B)
    h' : IsUnit A.det
    ⊢ Eq (HMul.hMul B (Inv.inv A)) 1
  -/
  rw [h]
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Eq (Inv.inv A) (Inv.inv B)
    h' : IsUnit A.det
    ⊢ Eq (HMul.hMul B (Inv.inv B)) 1
  -/
  refine mul_nonsing_inv _ ?_
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Eq (Inv.inv A) (Inv.inv B)
    h' : IsUnit A.det
    ⊢ IsUnit B.det
  -/
  rwa [← isUnit_nonsing_inv_det_iff, ← h, isUnit_nonsing_inv_det_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_zero : (0 : Matrix n n α)⁻¹ = 0 := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    ⊢ Eq (Inv.inv 0) 0
  -/
  cases' subsingleton_or_nontrivial α with ht ht
    /-
      case inl
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Subsingleton α
      ⊢ Eq (Inv.inv 0) 0
    -/
  · simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    ht : Nontrivial α
    ⊢ Eq (Inv.inv 0) 0
  -/
  rcases (Fintype.card n).zero_le.eq_or_lt with hc | hc
    /-
      case inr.inl
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc : Eq 0 (Fintype.card n)
      ⊢ Eq (Inv.inv 0) 0
    -/
  · rw [eq_comm, Fintype.card_eq_zero_iff] at hc
    /-
      case inr.inl
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc : IsEmpty n
      ⊢ Eq (Inv.inv 0) 0
    -/
    haveI := hc
    /-
      case inr.inl
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc this : IsEmpty n
      ⊢ Eq (Inv.inv 0) 0
    -/
    ext i
    /-
      case inr.inl.a
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc this : IsEmpty n
      i j✝ : n
      ⊢ Eq (Inv.inv 0 i j✝) (0 i j✝)
    -/
    exact (IsEmpty.false i).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc : LT.lt 0 (Fintype.card n)
      ⊢ Eq (Inv.inv 0) 0
    -/
  · have hn : Nonempty n := Fintype.card_pos_iff.mp hc
    /-
      case inr.inr
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc : LT.lt 0 (Fintype.card n)
      hn : Nonempty n
      ⊢ Eq (Inv.inv 0) 0
    -/
    refine nonsing_inv_apply_not_isUnit _ ?_
    /-
      case inr.inr
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      ht : Nontrivial α
      hc : LT.lt 0 (Fintype.card n)
      hn : Nonempty n
      ⊢ Not (IsUnit (Matrix.det 0))
    -/
    simp [hn]
    /-
      🎉 no goals
    -/


noncomputable instance : InvOneClass (Matrix n n α) :=
                                                               /-
                                                                 l : Type u_1
                                                                 m : Type u
                                                                 n : Type u'
                                                                 α : Type v
                                                                 inst✝² : Fintype n
                                                                 inst✝¹ : DecidableEq n
                                                                 inst✝ : CommRing α
                                                                 A B : Matrix n n α
                                                                 ⊢ Eq (HMul.hMul 1 1) 1
                                                               -/
  { Matrix.one, Matrix.inv with inv_one := inv_eq_left_inv (by simp) }
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem inv_smul (k : α) [Invertible k] (h : IsUnit A.det) : (k • A)⁻¹ = ⅟ k • A⁻¹ :=
                      /-
                        n : Type u'
                        α : Type v
                        inst✝³ : Fintype n
                        inst✝² : DecidableEq n
                        inst✝¹ : CommRing α
                        A : Matrix n n α
                        k : α
                        inst✝ : Invertible k
                        h : IsUnit A.det
                        ⊢ Eq (HMul.hMul (HSMul.hSMul (Invertible.invOf k) (Inv.inv A)) (HSMul.hSMul k  …
                      -/
  inv_eq_left_inv (by simp [h, smul_smul])
                      /-
                        🎉 no goals
                      -/


theorem inv_smul' (k : αˣ) (h : IsUnit A.det) : (k • A)⁻¹ = k⁻¹ • A⁻¹ :=
                      /-
                        n : Type u'
                        α : Type v
                        inst✝² : Fintype n
                        inst✝¹ : DecidableEq n
                        inst✝ : CommRing α
                        A : Matrix n n α
                        k : Units α
                        h : IsUnit A.det
                        ⊢ Eq (HMul.hMul (HSMul.hSMul (Inv.inv k) (Inv.inv A)) (HSMul.hSMul k A)) 1
                      -/
  inv_eq_left_inv (by simp [h, smul_smul])
                      /-
                        🎉 no goals
                      -/


theorem inv_adjugate (A : Matrix n n α) (h : IsUnit A.det) : (adjugate A)⁻¹ = h.unit⁻¹ • A := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (Inv.inv A.adjugate) (HSMul.hSMul (Inv.inv h.unit) A)
  -/
  refine inv_eq_left_inv ?_
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    h : IsUnit A.det
    ⊢ Eq (HMul.hMul (HSMul.hSMul (Inv.inv h.unit) A) A.adjugate) 1
  -/
  rw [smul_mul, mul_adjugate, Units.smul_def, smul_smul, h.val_inv_mul, one_smul]
  /-
    🎉 no goals
  -/


/-- `diagonal v` is invertible if `v` is -/
def diagonalInvertible {α} [NonAssocSemiring α] (v : n → α) [Invertible v] :
    Invertible (diagonal v) :=
  Invertible.map (diagonalRingHom n α) v


theorem invOf_diagonal_eq {α} [Semiring α] (v : n → α) [Invertible v] [Invertible (diagonal v)] :
    ⅟ (diagonal v) = diagonal (⅟ v) := by
  /-
    n : Type u'
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    α : Type u_2
    inst✝² : Semiring α
    v : n → α
    inst✝¹ : Invertible v
    inst✝ : Invertible (Matrix.diagonal v)
    ⊢ Eq (Invertible.invOf (Matrix.diagonal v)) (Matrix.diagonal (Invertible.invOf …
  -/
  letI := diagonalInvertible v
  -- Porting note: no longer need `haveI := Invertible.subsingleton (diagonal v)`
  /-
    n : Type u'
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    α : Type u_2
    inst✝² : Semiring α
    v : n → α
    inst✝¹ : Invertible v
    inst✝ : Invertible (Matrix.diagonal v)
    this : Invertible (Matrix.diagonal v) := Matrix.diagonalInvertible v
    ⊢ Eq (Invertible.invOf (Matrix.diagonal v)) (Matrix.diagonal (Invertible.invOf …
  -/
  convert (rfl : ⅟ (diagonal v) = _)
  /-
    🎉 no goals
  -/


/-- `v` is invertible if `diagonal v` is -/
def invertibleOfDiagonalInvertible (v : n → α) [Invertible (diagonal v)] : Invertible v where
  invOf := diag (⅟ (diagonal v))
  invOf_mul_self :=
    funext fun i => by
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.diagonal v)).diag v i) (1 i)
      -/
      letI : Invertible (diagonal v).det := detInvertibleOfInvertible _
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.diagonal v)).diag v i) (1 i)
      -/
      rw [invOf_eq, diag_smul, adjugate_diagonal, diag_diagonal]
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (HSMul.hSMul (Invertible.invOf (Matrix.diagonal v).det) fun i  …
      -/
      dsimp
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (HMul.hMul (Invertible.invOf (Matrix.diagonal v).det) ((Finset …
      -/
      rw [mul_assoc, prod_erase_mul _ _ (Finset.mem_univ _), ← det_diagonal]
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.diagonal v).det) (Matrix.diagonal v) …
      -/
      exact mul_invOf_self _
      /-
        🎉 no goals
      -/
  mul_invOf_self :=
    funext fun i => by
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        ⊢ Eq (HMul.hMul v (Invertible.invOf (Matrix.diagonal v)).diag i) (1 i)
      -/
      letI : Invertible (diagonal v).det := detInvertibleOfInvertible _
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul v (Invertible.invOf (Matrix.diagonal v)).diag i) (1 i)
      -/
      rw [invOf_eq, diag_smul, adjugate_diagonal, diag_diagonal]
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul v (HSMul.hSMul (Invertible.invOf (Matrix.diagonal v).det) fun  …
      -/
      dsimp
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (v i) (HMul.hMul (Invertible.invOf (Matrix.diagonal v).det) (( …
      -/
      rw [mul_left_comm, mul_prod_erase _ _ (Finset.mem_univ _), ← det_diagonal]
      /-
        l : Type u_1
        m : Type u
        n : Type u'
        α : Type v
        inst✝³ : Fintype n
        inst✝² : DecidableEq n
        inst✝¹ : CommRing α
        A B : Matrix n n α
        v : n → α
        inst✝ : Invertible (Matrix.diagonal v)
        i : n
        this : Invertible (Matrix.diagonal v).det := (Matrix.diagonal v).detInvertible …
        ⊢ Eq (HMul.hMul (Invertible.invOf (Matrix.diagonal v).det) (Matrix.diagonal v) …
      -/
      exact mul_invOf_self _
      /-
        🎉 no goals
      -/


/-- Together `Matrix.diagonalInvertible` and `Matrix.invertibleOfDiagonalInvertible` form an
equivalence, although both sides of the equiv are subsingleton anyway. -/
@[simps]
def diagonalInvertibleEquivInvertible (v : n → α) : Invertible (diagonal v) ≃ Invertible v where
  toFun := @invertibleOfDiagonalInvertible _ _ _ _ _ _
  invFun := @diagonalInvertible _ _ _ _ _ _
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- When lowered to a prop, `Matrix.diagonalInvertibleEquivInvertible` forms an `iff`. -/
@[simp]
theorem isUnit_diagonal {v : n → α} : IsUnit (diagonal v) ↔ IsUnit v := by
  simp only [← nonempty_invertible_iff_isUnit,
    (diagonalInvertibleEquivInvertible v).nonempty_congr]


theorem inv_diagonal (v : n → α) : (diagonal v)⁻¹ = diagonal (Ring.inverse v) := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    v : n → α
    ⊢ Eq (Inv.inv (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
  -/
  rw [nonsing_inv_eq_ring_inverse]
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    v : n → α
    ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
  -/
  by_cases h : IsUnit v
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : IsUnit v
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
  · have := isUnit_diagonal.mpr h
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : IsUnit v
      this : IsUnit (Matrix.diagonal v)
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
    cases this.nonempty_invertible
    /-
      case pos.intro
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : IsUnit v
      this : IsUnit (Matrix.diagonal v)
      val✝ : Invertible (Matrix.diagonal v)
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
    cases h.nonempty_invertible
    /-
      case pos.intro.intro
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : IsUnit v
      this : IsUnit (Matrix.diagonal v)
      val✝¹ : Invertible (Matrix.diagonal v)
      val✝ : Invertible v
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
    rw [Ring.inverse_invertible, Ring.inverse_invertible, invOf_diagonal_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : Not (IsUnit v)
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
  · have := isUnit_diagonal.not.mpr h
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      v : n → α
      h : Not (IsUnit v)
      this : Not (IsUnit (Matrix.diagonal v))
      ⊢ Eq (Ring.inverse (Matrix.diagonal v)) (Matrix.diagonal (Ring.inverse v))
    -/
    rw [Ring.inverse_non_unit _ h, Pi.zero_def, diagonal_zero, Ring.inverse_non_unit _ this]
    /-
      🎉 no goals
    -/


/-- The inverse of a 1×1 or 0×0 matrix is always diagonal.

While we could write this as `of fun _ _ => Ring.inverse (A default default)` on the RHS, this is
less useful because:

* It wouldn't work for 0×0 matrices.
* More things are true about diagonal matrices than constant matrices, and so more lemmas exist.

`Matrix.diagonal_unique` can be used to reach this form, while `Ring.inverse_eq_inv` can be used
to replace `Ring.inverse` with `⁻¹`.
-/
@[simp]
theorem inv_subsingleton [Subsingleton m] [Fintype m] [DecidableEq m] (A : Matrix m m α) :
    A⁻¹ = diagonal fun i => Ring.inverse (A i i) := by
  /-
    m : Type u
    α : Type v
    inst✝³ : CommRing α
    inst✝² : Subsingleton m
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m α
    ⊢ Eq (Inv.inv A) (Matrix.diagonal fun i => Ring.inverse (A i i))
  -/
  rw [inv_def, adjugate_subsingleton, smul_one_eq_diagonal]
  /-
    m : Type u
    α : Type v
    inst✝³ : CommRing α
    inst✝² : Subsingleton m
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m α
    ⊢ Eq (Matrix.diagonal fun x => Ring.inverse A.det) (Matrix.diagonal fun i => R …
  -/
  congr! with i
  /-
    case h.e'_5.h.h.e'_3
    m : Type u
    α : Type v
    inst✝³ : CommRing α
    inst✝² : Subsingleton m
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m α
    i : m
    ⊢ Eq A.det (A i i)
  -/
  exact det_eq_elem_of_subsingleton _ _
  /-
    🎉 no goals
  -/


/-- The **Woodbury Identity** (`⁻¹` version). -/
theorem add_mul_mul_inv_eq_sub (hA : IsUnit A) (hC : IsUnit C) (hAC : IsUnit (C⁻¹ + V * A⁻¹ * U)) :
    (A + U * C * V)⁻¹ = A⁻¹ - A⁻¹ * U * (C⁻¹ + V * A⁻¹ * U)⁻¹ * V * A⁻¹ := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  obtain ⟨_⟩ := hA.nonempty_invertible
  /-
    case intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝ : Invertible A
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  obtain ⟨_⟩ := hC.nonempty_invertible
  /-
    case intro.intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝¹ : Invertible A
    val✝ : Invertible C
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  obtain ⟨iAC⟩ := hAC.nonempty_invertible
  /-
    case intro.intro.intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝¹ : Invertible A
    val✝ : Invertible C
    iAC : Invertible (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  simp only [← invOf_eq_nonsing_inv] at iAC
  /-
    case intro.intro.intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝¹ : Invertible A
    val✝ : Invertible C
    iAC : Invertible (HAdd.hAdd (Invertible.invOf C) (HMul.hMul (HMul.hMul V (Inve …
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  letI := invertibleAddMulMul A U C V
  /-
    case intro.intro.intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝¹ : Invertible A
    val✝ : Invertible C
    iAC : Invertible (HAdd.hAdd (Invertible.invOf C) (HMul.hMul (HMul.hMul V (Inve …
    this : Invertible (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V)) := A.invertibleA …
    ⊢ Eq (Inv.inv (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub (Inv.inv …
  -/
  simp only [← invOf_eq_nonsing_inv]
  /-
    case intro.intro.intro
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix n n α
    U : Matrix n m α
    C : Matrix m m α
    V : Matrix m n α
    hA : IsUnit A
    hC : IsUnit C
    hAC : IsUnit (HAdd.hAdd (Inv.inv C) (HMul.hMul (HMul.hMul V (Inv.inv A)) U))
    val✝¹ : Invertible A
    val✝ : Invertible C
    iAC : Invertible (HAdd.hAdd (Invertible.invOf C) (HMul.hMul (HMul.hMul V (Inve …
    this : Invertible (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V)) := A.invertibleA …
    ⊢ Eq (Invertible.invOf (HAdd.hAdd A (HMul.hMul (HMul.hMul U C) V))) (HSub.hSub …
  -/
  apply invOf_add_mul_mul
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_inv_inv (A : Matrix n n α) : A⁻¹⁻¹⁻¹ = A⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (Inv.inv (Inv.inv (Inv.inv A))) (Inv.inv A)
  -/
  by_cases h : IsUnit A.det
    /-
      case pos
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : IsUnit A.det
      ⊢ Eq (Inv.inv (Inv.inv (Inv.inv A))) (Inv.inv A)
    -/
  · rw [nonsing_inv_nonsing_inv _ h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u'
      α : Type v
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommRing α
      A : Matrix n n α
      h : Not (IsUnit A.det)
      ⊢ Eq (Inv.inv (Inv.inv (Inv.inv A))) (Inv.inv A)
    -/
  · simp [nonsing_inv_apply_not_isUnit _ h]
    /-
      🎉 no goals
    -/


/-- The `Matrix` version of `inv_add_inv'` -/
theorem inv_add_inv {A B : Matrix n n α} (h : IsUnit A ↔ IsUnit B) :
    A⁻¹ + B⁻¹ = A⁻¹ * (A + B) * B⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Iff (IsUnit A) (IsUnit B)
    ⊢ Eq (HAdd.hAdd (Inv.inv A) (Inv.inv B)) (HMul.hMul (HMul.hMul (Inv.inv A) (HA …
  -/
  simpa only [nonsing_inv_eq_ring_inverse] using Ring.inverse_add_inverse h
  /-
    🎉 no goals
  -/


/-- The `Matrix` version of `inv_sub_inv'` -/
theorem inv_sub_inv {A B : Matrix n n α} (h : IsUnit A ↔ IsUnit B) :
    A⁻¹ - B⁻¹ = A⁻¹ * (B - A) * B⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    h : Iff (IsUnit A) (IsUnit B)
    ⊢ Eq (HSub.hSub (Inv.inv A) (Inv.inv B)) (HMul.hMul (HMul.hMul (Inv.inv A) (HS …
  -/
  simpa only [nonsing_inv_eq_ring_inverse] using Ring.inverse_sub_inverse h
  /-
    🎉 no goals
  -/


theorem mul_inv_rev (A B : Matrix n n α) : (A * B)⁻¹ = B⁻¹ * A⁻¹ := by
  /-
    n : Type u'
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ Eq (Inv.inv (HMul.hMul A B)) (HMul.hMul (Inv.inv B) (Inv.inv A))
  -/
  simp only [inv_def]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, det_mul, adjugate_mul_distrib,
    Ring.mul_inverse_rev]


/-- A version of `List.prod_inv_reverse` for `Matrix.inv`. -/
theorem list_prod_inv_reverse : ∀ l : List (Matrix n n α), l.prod⁻¹ = (l.reverse.map Inv.inv).prod
             /-
               n : Type u'
               α : Type v
               inst✝² : Fintype n
               inst✝¹ : DecidableEq n
               inst✝ : CommRing α
               ⊢ Eq (Inv.inv List.nil.prod) (List.map Inv.inv List.nil.reverse).prod
             -/
  | [] => by rw [List.reverse_nil, List.map_nil, List.prod_nil, inv_one]
             /-
               🎉 no goals
             -/
  | A::Xs => by
    rw [List.reverse_cons', List.map_concat, List.prod_concat, List.prod_cons,
      mul_inv_rev, list_prod_inv_reverse Xs]


/-- One form of **Cramer's rule**. See `Matrix.mulVec_cramer` for a stronger form. -/
@[simp]
theorem det_smul_inv_mulVec_eq_cramer (A : Matrix n n α) (b : n → α) (h : IsUnit A.det) :
    A.det • A⁻¹ *ᵥ b = cramer A b := by
  rw [cramer_eq_adjugate_mulVec, A.nonsing_inv_apply h, ← smul_mulVec_assoc, smul_smul,
    h.mul_val_inv, one_smul]


/-- One form of **Cramer's rule**. See `Matrix.mulVec_cramer` for a stronger form. -/
@[simp]
theorem det_smul_inv_vecMul_eq_cramer_transpose (A : Matrix n n α) (b : n → α) (h : IsUnit A.det) :
    A.det • b ᵥ* A⁻¹ = cramer Aᵀ b := by
  rw [← A⁻¹.transpose_transpose, vecMul_transpose, transpose_nonsing_inv, ← det_transpose,
    Aᵀ.det_smul_inv_mulVec_eq_cramer _ (isUnit_det_transpose A h)]


/-- `A.submatrix e₁ e₂` is invertible if `A` is -/
def submatrixEquivInvertible (A : Matrix m m α) (e₁ e₂ : n ≃ m) [Invertible A] :
    Invertible (A.submatrix e₁ e₂) :=
  invertibleOfRightInverse _ ((⅟ A).submatrix e₂ e₁) <| by
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁵ : Fintype n
      inst✝⁴ : DecidableEq n
      inst✝³ : CommRing α
      A✝ B : Matrix n n α
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      inst✝ : Invertible A
      ⊢ Eq (HMul.hMul (A.submatrix ⇑e₁ ⇑e₂) ((Invertible.invOf A).submatrix ⇑e₂ ⇑e₁) …
    -/
    rw [Matrix.submatrix_mul_equiv, mul_invOf_self, submatrix_one_equiv]
    /-
      🎉 no goals
    -/


/-- `A` is invertible if `A.submatrix e₁ e₂` is -/
def invertibleOfSubmatrixEquivInvertible (A : Matrix m m α) (e₁ e₂ : n ≃ m)
    [Invertible (A.submatrix e₁ e₂)] : Invertible A :=
  invertibleOfRightInverse _ ((⅟ (A.submatrix e₁ e₂)).submatrix e₂.symm e₁.symm) <| by
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁵ : Fintype n
      inst✝⁴ : DecidableEq n
      inst✝³ : CommRing α
      A✝ B : Matrix n n α
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      inst✝ : Invertible (A.submatrix ⇑e₁ ⇑e₂)
      ⊢ Eq (HMul.hMul A ((Invertible.invOf (A.submatrix ⇑e₁ ⇑e₂)).submatrix ⇑e₂.symm …
    -/
    have : A = (A.submatrix e₁ e₂).submatrix e₁.symm e₂.symm := by simp
    -- Porting note: was
    -- conv in _ * _ =>
    --   congr
    --   rw [this]
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁵ : Fintype n
      inst✝⁴ : DecidableEq n
      inst✝³ : CommRing α
      A✝ B : Matrix n n α
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      inst✝ : Invertible (A.submatrix ⇑e₁ ⇑e₂)
      this : Eq A ((A.submatrix ⇑e₁ ⇑e₂).submatrix ⇑e₁.symm ⇑e₂.symm)
      ⊢ Eq (HMul.hMul A ((Invertible.invOf (A.submatrix ⇑e₁ ⇑e₂)).submatrix ⇑e₂.symm …
    -/
    rw [congr_arg₂ (· * ·) this rfl]
    /-
      l : Type u_1
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁵ : Fintype n
      inst✝⁴ : DecidableEq n
      inst✝³ : CommRing α
      A✝ B : Matrix n n α
      inst✝² : Fintype m
      inst✝¹ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      inst✝ : Invertible (A.submatrix ⇑e₁ ⇑e₂)
      this : Eq A ((A.submatrix ⇑e₁ ⇑e₂).submatrix ⇑e₁.symm ⇑e₂.symm)
      ⊢ Eq (HMul.hMul ((A.submatrix ⇑e₁ ⇑e₂).submatrix ⇑e₁.symm ⇑e₂.symm) ((Invertib …
    -/
    rw [Matrix.submatrix_mul_equiv, mul_invOf_self, submatrix_one_equiv]
    /-
      🎉 no goals
    -/


theorem invOf_submatrix_equiv_eq (A : Matrix m m α) (e₁ e₂ : n ≃ m) [Invertible A]
    [Invertible (A.submatrix e₁ e₂)] : ⅟ (A.submatrix e₁ e₂) = (⅟ A).submatrix e₂ e₁ := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommRing α
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    A : Matrix m m α
    e₁ e₂ : Equiv n m
    inst✝¹ : Invertible A
    inst✝ : Invertible (A.submatrix ⇑e₁ ⇑e₂)
    ⊢ Eq (Invertible.invOf (A.submatrix ⇑e₁ ⇑e₂)) ((Invertible.invOf A).submatrix  …
  -/
  letI := submatrixEquivInvertible A e₁ e₂
  -- Porting note: no longer need `haveI := Invertible.subsingleton (A.submatrix e₁ e₂)`
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommRing α
    inst✝³ : Fintype m
    inst✝² : DecidableEq m
    A : Matrix m m α
    e₁ e₂ : Equiv n m
    inst✝¹ : Invertible A
    inst✝ : Invertible (A.submatrix ⇑e₁ ⇑e₂)
    this : Invertible (A.submatrix ⇑e₁ ⇑e₂) := A.submatrixEquivInvertible e₁ e₂
    ⊢ Eq (Invertible.invOf (A.submatrix ⇑e₁ ⇑e₂)) ((Invertible.invOf A).submatrix  …
  -/
  convert (rfl : ⅟ (A.submatrix e₁ e₂) = _)
  /-
    🎉 no goals
  -/


/-- Together `Matrix.submatrixEquivInvertible` and
`Matrix.invertibleOfSubmatrixEquivInvertible` form an equivalence, although both sides of the
equiv are subsingleton anyway. -/
@[simps]
def submatrixEquivInvertibleEquivInvertible (A : Matrix m m α) (e₁ e₂ : n ≃ m) :
    Invertible (A.submatrix e₁ e₂) ≃ Invertible A where
  toFun _ := invertibleOfSubmatrixEquivInvertible A e₁ e₂
  invFun _ := submatrixEquivInvertible A e₁ e₂
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- When lowered to a prop, `Matrix.invertibleOfSubmatrixEquivInvertible` forms an `iff`. -/
@[simp]
theorem isUnit_submatrix_equiv {A : Matrix m m α} (e₁ e₂ : n ≃ m) :
    IsUnit (A.submatrix e₁ e₂) ↔ IsUnit A := by
  simp only [← nonempty_invertible_iff_isUnit,
    (submatrixEquivInvertibleEquivInvertible A _ _).nonempty_congr]


@[simp]
theorem inv_submatrix_equiv (A : Matrix m m α) (e₁ e₂ : n ≃ m) :
    (A.submatrix e₁ e₂)⁻¹ = A⁻¹.submatrix e₂ e₁ := by
  /-
    m : Type u
    n : Type u'
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommRing α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m α
    e₁ e₂ : Equiv n m
    ⊢ Eq (Inv.inv (A.submatrix ⇑e₁ ⇑e₂)) ((Inv.inv A).submatrix ⇑e₂ ⇑e₁)
  -/
  by_cases h : IsUnit A
    /-
      case pos
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      h : IsUnit A
      ⊢ Eq (Inv.inv (A.submatrix ⇑e₁ ⇑e₂)) ((Inv.inv A).submatrix ⇑e₂ ⇑e₁)
    -/
  · cases h.nonempty_invertible
    /-
      case pos.intro
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      h : IsUnit A
      val✝ : Invertible A
      ⊢ Eq (Inv.inv (A.submatrix ⇑e₁ ⇑e₂)) ((Inv.inv A).submatrix ⇑e₂ ⇑e₁)
    -/
    letI := submatrixEquivInvertible A e₁ e₂
    /-
      case pos.intro
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      h : IsUnit A
      val✝ : Invertible A
      this : Invertible (A.submatrix ⇑e₁ ⇑e₂) := A.submatrixEquivInvertible e₁ e₂
      ⊢ Eq (Inv.inv (A.submatrix ⇑e₁ ⇑e₂)) ((Inv.inv A).submatrix ⇑e₂ ⇑e₁)
    -/
    rw [← invOf_eq_nonsing_inv, ← invOf_eq_nonsing_inv, invOf_submatrix_equiv_eq A]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Type u
      n : Type u'
      α : Type v
      inst✝⁴ : Fintype n
      inst✝³ : DecidableEq n
      inst✝² : CommRing α
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      A : Matrix m m α
      e₁ e₂ : Equiv n m
      h : Not (IsUnit A)
      ⊢ Eq (Inv.inv (A.submatrix ⇑e₁ ⇑e₂)) ((Inv.inv A).submatrix ⇑e₂ ⇑e₁)
    -/
  · have := (isUnit_submatrix_equiv e₁ e₂).not.mpr h
    simp_rw [nonsing_inv_eq_ring_inverse, Ring.inverse_non_unit _ h, Ring.inverse_non_unit _ this,
      submatrix_zero, Pi.zero_apply]


theorem inv_reindex (e₁ e₂ : n ≃ m) (A : Matrix n n α) : (reindex e₁ e₂ A)⁻¹ = reindex e₂ e₁ A⁻¹ :=
  inv_submatrix_equiv A e₁.symm e₂.symm


/-- A variant of `Matrix.det_units_conj`. -/
theorem det_conj {M : Matrix m m α} (h : IsUnit M) (N : Matrix m m α) :
                                    /-
                                      m : Type u
                                      α : Type v
                                      inst✝² : CommRing α
                                      inst✝¹ : Fintype m
                                      inst✝ : DecidableEq m
                                      M : Matrix m m α
                                      h : IsUnit M
                                      N : Matrix m m α
                                      ⊢ Eq (HMul.hMul (HMul.hMul M N) (Inv.inv M)).det N.det
                                    -/
    det (M * N * M⁻¹) = det N := by rw [← h.unit_spec, ← coe_units_inv, det_units_conj]
                                    /-
                                      🎉 no goals
                                    -/


/-- A variant of `Matrix.det_units_conj'`. -/
theorem det_conj' {M : Matrix m m α} (h : IsUnit M) (N : Matrix m m α) :
                                    /-
                                      m : Type u
                                      α : Type v
                                      inst✝² : CommRing α
                                      inst✝¹ : Fintype m
                                      inst✝ : DecidableEq m
                                      M : Matrix m m α
                                      h : IsUnit M
                                      N : Matrix m m α
                                      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv M) N) M).det N.det
                                    -/
    det (M⁻¹ * N * M) = det N := by rw [← h.unit_spec, ← coe_units_inv, det_units_conj']
                                    /-
                                      🎉 no goals
                                    -/


/-- A variant of `Matrix.trace_units_conj`. -/
theorem trace_conj {M : Matrix m m α} (h : IsUnit M) (N : Matrix m m α) :
                                        /-
                                          m : Type u
                                          α : Type v
                                          inst✝² : CommRing α
                                          inst✝¹ : Fintype m
                                          inst✝ : DecidableEq m
                                          M : Matrix m m α
                                          h : IsUnit M
                                          N : Matrix m m α
                                          ⊢ Eq (HMul.hMul (HMul.hMul M N) (Inv.inv M)).trace N.trace
                                        -/
    trace (M * N * M⁻¹) = trace N := by rw [← h.unit_spec, ← coe_units_inv, trace_units_conj]
                                        /-
                                          🎉 no goals
                                        -/


/-- A variant of `Matrix.trace_units_conj'`. -/
theorem trace_conj' {M : Matrix m m α} (h : IsUnit M) (N : Matrix m m α) :
                                        /-
                                          m : Type u
                                          α : Type v
                                          inst✝² : CommRing α
                                          inst✝¹ : Fintype m
                                          inst✝ : DecidableEq m
                                          M : Matrix m m α
                                          h : IsUnit M
                                          N : Matrix m m α
                                          ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv M) N) M).trace N.trace
                                        -/
    trace (M⁻¹ * N * M) = trace N := by rw [← h.unit_spec, ← coe_units_inv, trace_units_conj']
                                        /-
                                          🎉 no goals
                                        -/


