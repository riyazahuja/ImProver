/-- The isometry between a weighted sum of squares on the complex numbers and the
sum of squares, i.e. `weightedSumSquares` with weights 1 or 0. -/
noncomputable def isometryEquivSumSquares (w' : ι → ℂ) :
    IsometryEquiv (weightedSumSquares ℂ w')
      (weightedSumSquares ℂ (fun i => if w' i = 0 then 0 else 1 : ι → ℂ)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    ⊢ (QuadraticMap.weightedSumSquares Complex w').IsometryEquiv (QuadraticMap.wei …
  -/
  let w i := if h : w' i = 0 then (1 : Units ℂ) else Units.mk0 (w' i) h
  have hw' : ∀ i : ι, (w i : ℂ) ^ (-(1 / 2 : ℂ)) ≠ 0 := by
    intro i hi
    exact (w i).ne_zero ((Complex.cpow_eq_zero_iff _ _).1 hi).1
  convert QuadraticMap.isometryEquivBasisRepr (weightedSumSquares ℂ w')
    ((Pi.basisFun ℂ ι).unitsSMul fun i => (isUnit_iff_ne_zero.2 <| hw' i).unit)
  /-
    case h.e'_13
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    ⊢ Eq (QuadraticMap.weightedSumSquares Complex fun i => ite (Eq (w' i) 0) 0 1)  …
  -/
  ext1 v
  /-
    case h.e'_13.H
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    ⊢ Eq ((QuadraticMap.weightedSumSquares Complex fun i => ite (Eq (w' i) 0) 0 1) …
  -/
  rw [basisRepr_apply, weightedSumSquares_apply, weightedSumSquares_apply]
  /-
    case h.e'_13.H
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (ite (Eq (w' i) 0) 0 1) (HMul.hMul  …
  -/
  refine sum_congr rfl fun j hj => ?_
  have hsum : (∑ i : ι, v i • ((isUnit_iff_ne_zero.2 <| hw' i).unit : ℂ) • (Pi.basisFun ℂ ι) i) j =
      v j • w j ^ (-(1 / 2 : ℂ)) := by
    classical
    rw [Finset.sum_apply, sum_eq_single j, Pi.basisFun_apply, IsUnit.unit_spec,
      Pi.smul_apply, Pi.smul_apply, Pi.single_eq_same, smul_eq_mul,
      smul_eq_mul, smul_eq_mul, mul_one]
    · intro i _ hij
      rw [Pi.basisFun_apply, Pi.smul_apply, Pi.smul_apply,
        Pi.single_eq_of_ne hij.symm, smul_eq_mul, smul_eq_mul,
        mul_zero, mul_zero]
    intro hj'; exact False.elim (hj' hj)
  /-
    case h.e'_13.H
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    ⊢ Eq (HSMul.hSMul (ite (Eq (w' j) 0) 0 1) (HMul.hMul (v j) (v j))) (HSMul.hSMu …
  -/
  simp_rw [Basis.unitsSMul_apply]
  /-
    case h.e'_13.H
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    ⊢ Eq (HSMul.hSMul (ite (Eq (w' j) 0) 0 1) (HMul.hMul (v j) (v j))) (HSMul.hSMu …
  -/
  erw [hsum, smul_eq_mul]
  /-
    case h.e'_13.H
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    ⊢ Eq (HMul.hMul (ite (Eq (w' j) 0) 0 1) (HMul.hMul (v j) (v j))) (HSMul.hSMul  …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      inst✝ : Fintype ι
      w' : ι → Complex
      w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
      hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
      v : ι → Complex
      j : ι
      hj : Membership.mem Finset.univ j
      hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
      h : Eq (w' j) 0
      ⊢ Eq (HMul.hMul 0 (HMul.hMul (v j) (v j))) (HSMul.hSMul (w' j) (HMul.hMul (HSM …
    -/
  · simp only [h, zero_smul, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    h : Not (Eq (w' j) 0)
    ⊢ Eq (HMul.hMul 1 (HMul.hMul (v j) (v j))) (HSMul.hSMul (w' j) (HMul.hMul (HSM …
  -/
  have hww' : w' j = w j := by simp only [w, dif_neg h, Units.val_mk0]
  /-
    case neg
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    h : Not (Eq (w' j) 0)
    hww' : Eq (w' j) ↑(w j)
    ⊢ Eq (HMul.hMul 1 (HMul.hMul (v j) (v j))) (HSMul.hSMul (w' j) (HMul.hMul (HSM …
  -/
  simp (config := {zeta := false}) only [one_mul, Units.val_mk0, smul_eq_mul]
  /-
    case neg
    ι : Type u_1
    inst✝ : Fintype ι
    w' : ι → Complex
    w : ι → Units Complex := fun i => dite (Eq (w' i) 0) (fun h => 1) fun h => Uni …
    hw' : ∀ (i : ι), Ne (HPow.hPow (↑(w i)) (Neg.neg (1 / 2))) 0
    v : ι → Complex
    j : ι
    hj : Membership.mem Finset.univ j
    hsum : Eq (Finset.univ.sum (fun i => HSMul.hSMul (v i) (HSMul.hSMul (↑⋯.unit)  …
    h : Not (Eq (w' j) 0)
    hww' : Eq (w' j) ↑(w j)
    ⊢ Eq (HMul.hMul (v j) (v j)) (HMul.hMul (w' j) (HMul.hMul (HMul.hMul (v j) (HP …
  -/
  rw [hww']
  suffices v j * v j = w j ^ (-(1 / 2 : ℂ)) * w j ^ (-(1 / 2 : ℂ)) * w j * v j * v j by
    rw [this]; ring
  rw [← Complex.cpow_add _ _ (w j).ne_zero, show -(1 / 2 : ℂ) + -(1 / 2) = -1 by simp [← two_mul],
    Complex.cpow_neg_one, inv_mul_cancel₀ (w j).ne_zero, one_mul]


/-- The isometry between a weighted sum of squares on the complex numbers and the
sum of squares, i.e. `weightedSumSquares` with weight `fun (i : ι) => 1`. -/
noncomputable def isometryEquivSumSquaresUnits (w : ι → Units ℂ) :
    IsometryEquiv (weightedSumSquares ℂ w) (weightedSumSquares ℂ (1 : ι → ℂ)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    w : ι → Units Complex
    ⊢ (QuadraticMap.weightedSumSquares Complex w).IsometryEquiv (QuadraticMap.weig …
  -/
  simpa using isometryEquivSumSquares ((↑) ∘ w)
  /-
    🎉 no goals
  -/


/-- A nondegenerate quadratic form on the complex numbers is equivalent to
the sum of squares, i.e. `weightedSumSquares` with weight `fun (i : ι) => 1`. -/
theorem equivalent_sum_squares {M : Type*} [AddCommGroup M] [Module ℂ M] [FiniteDimensional ℂ M]
    (Q : QuadraticForm ℂ M) (hQ : (associated (R := ℂ) Q).SeparatingLeft) :
    Equivalent Q (weightedSumSquares ℂ (1 : Fin (Module.finrank ℂ M) → ℂ)) :=
  let ⟨w, ⟨hw₁⟩⟩ := Q.equivalent_weightedSumSquares_units_of_nondegenerate' hQ
  ⟨hw₁.trans (isometryEquivSumSquaresUnits w)⟩


/-- All nondegenerate quadratic forms on the complex numbers are equivalent. -/
theorem complex_equivalent {M : Type*} [AddCommGroup M] [Module ℂ M] [FiniteDimensional ℂ M]
    (Q₁ Q₂ : QuadraticForm ℂ M) (hQ₁ : (associated (R := ℂ) Q₁).SeparatingLeft)
    (hQ₂ : (associated (R := ℂ) Q₂).SeparatingLeft) : Equivalent Q₁ Q₂ :=
  (Q₁.equivalent_sum_squares hQ₁).trans (Q₂.equivalent_sum_squares hQ₂).symm


