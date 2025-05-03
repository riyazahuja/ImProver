/-- Auxiliary function used for computing inverse of a power series -/
protected def inv.aux : R → R⟦X⟧ → R⟦X⟧ :=
  MvPowerSeries.inv.aux


theorem coeff_inv_aux (n : ℕ) (a : R) (φ : R⟦X⟧) :
    coeff R n (inv.aux a φ) =
      if n = 0 then a
      else
        -a *
          ∑ x ∈ antidiagonal n,
            if x.2 < n then coeff R x.1 φ * coeff R x.2 (inv.aux a φ) else 0 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : R
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R n) (PowerSeries.inv.aux a φ)) (ite (Eq n 0) a (HMul …
  -/
  rw [coeff, inv.aux, MvPowerSeries.coeff_inv_aux]
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : R
    φ : PowerSeries R
    ⊢ Eq (ite (Eq (Finsupp.single Unit.unit n) 0) a (HMul.hMul (Neg.neg a) ((Finse …
  -/
  simp only [Finsupp.single_eq_zero]
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : R
    φ : PowerSeries R
    ⊢ Eq (ite (Eq n 0) a (HMul.hMul (Neg.neg a) ((Finset.HasAntidiagonal.antidiago …
  -/
  split_ifs; · rfl
               /-
                 🎉 no goals
               -/
  /-
    case neg
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : R
    φ : PowerSeries R
    h✝ : Not (Eq n 0)
    ⊢ Eq (HMul.hMul (Neg.neg a) ((Finset.HasAntidiagonal.antidiagonal (Finsupp.sin …
  -/
  congr 1
  /-
    case neg.e_a
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : R
    φ : PowerSeries R
    h✝ : Not (Eq n 0)
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (Finsupp.single Unit.unit n)).sum f …
  -/
  symm
  apply Finset.sum_nbij' (fun (a, b) ↦ (single () a, single () b))
    fun (f, g) ↦ (f (), g ())
    /-
      case neg.e_a.hi
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      ⊢ ∀ (a : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case neg.e_a.hj
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      ⊢ ∀ (a : Prod (Finsupp Unit Nat) (Finsupp Unit Nat)), Membership.mem (Finset.H …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case neg.e_a.left_inv
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      ⊢ ∀ (a : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case neg.e_a.right_inv
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      ⊢ ∀ (a : Prod (Finsupp Unit Nat) (Finsupp Unit Nat)), Membership.mem (Finset.H …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case neg.e_a.h
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      ⊢ ∀ (a_1 : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal  …
    -/
  · rintro ⟨i, j⟩ _hij
    /-
      case neg.e_a.h.mk
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      i j : Nat
      _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
      ⊢ Eq (ite (LT.lt { fst := i, snd := j }.2 n) (HMul.hMul ((PowerSeries.coeff R  …
    -/
    obtain H | H := le_or_lt n j
      /-
        case neg.e_a.h.mk.inl
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        a : R
        φ : PowerSeries R
        h✝ : Not (Eq n 0)
        i j : Nat
        _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
        H : LE.le n j
        ⊢ Eq (ite (LT.lt { fst := i, snd := j }.2 n) (HMul.hMul ((PowerSeries.coeff R  …
      -/
    · aesop
      /-
        🎉 no goals
      -/
    /-
      case neg.e_a.h.mk.inr
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      i j : Nat
      _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
      H : LT.lt j n
      ⊢ Eq (ite (LT.lt { fst := i, snd := j }.2 n) (HMul.hMul ((PowerSeries.coeff R  …
    -/
    rw [if_pos H, if_pos]
      /-
        case neg.e_a.h.mk.inr
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        a : R
        φ : PowerSeries R
        h✝ : Not (Eq n 0)
        i j : Nat
        _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
        H : LT.lt j n
        ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case neg.e_a.h.mk.inr.hc
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      a : R
      φ : PowerSeries R
      h✝ : Not (Eq n 0)
      i j : Nat
      _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
      H : LT.lt j n
      ⊢ LT.lt (PowerSeries.coeff_inv_aux.match_1 (fun x => Prod (Finsupp Unit Nat) ( …
    -/
    refine ⟨?_, fun hh ↦ H.not_le ?_⟩
      /-
        case neg.e_a.h.mk.inr.hc.refine_1
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        a : R
        φ : PowerSeries R
        h✝ : Not (Eq n 0)
        i j : Nat
        _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
        H : LT.lt j n
        ⊢ LE.le (PowerSeries.coeff_inv_aux.match_1 (fun x => Prod (Finsupp Unit Nat) ( …
      -/
    · rintro ⟨⟩
      /-
        case neg.e_a.h.mk.inr.hc.refine_1.unit
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        a : R
        φ : PowerSeries R
        h✝ : Not (Eq n 0)
        i j : Nat
        _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
        H : LT.lt j n
        ⊢ LE.le ((PowerSeries.coeff_inv_aux.match_1 (fun x => Prod (Finsupp Unit Nat)  …
      -/
      simpa [Finsupp.single_eq_same] using le_of_lt H
      /-
        🎉 no goals
      -/
      /-
        case neg.e_a.h.mk.inr.hc.refine_2
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        a : R
        φ : PowerSeries R
        h✝ : Not (Eq n 0)
        i j : Nat
        _hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd  …
        H : LT.lt j n
        hh : LE.le (Finsupp.single Unit.unit n) (PowerSeries.coeff_inv_aux.match_1 (fu …
        ⊢ LE.le n j
      -/
    · simpa [Finsupp.single_eq_same] using hh ()
      /-
        🎉 no goals
      -/


/-- A formal power series is invertible if the constant coefficient is invertible. -/
def invOfUnit (φ : R⟦X⟧) (u : Rˣ) : R⟦X⟧ :=
  MvPowerSeries.invOfUnit φ u


theorem coeff_invOfUnit (n : ℕ) (φ : R⟦X⟧) (u : Rˣ) :
    coeff R n (invOfUnit φ u) =
      if n = 0 then ↑u⁻¹
      else
        -↑u⁻¹ *
          ∑ x ∈ antidiagonal n,
            if x.2 < n then coeff R x.1 φ * coeff R x.2 (invOfUnit φ u) else 0 :=
  coeff_inv_aux n (↑u⁻¹ : R) φ


@[simp]
theorem constantCoeff_invOfUnit (φ : R⟦X⟧) (u : Rˣ) :
    constantCoeff R (invOfUnit φ u) = ↑u⁻¹ := by
  /-
    R : Type u_1
    inst✝ : Ring R
    φ : PowerSeries R
    u : Units R
    ⊢ Eq ((PowerSeries.constantCoeff R) (φ.invOfUnit u)) ↑(Inv.inv u)
  -/
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_invOfUnit, if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_invOfUnit (φ : R⟦X⟧) (u : Rˣ) (h : constantCoeff R φ = u) :
    φ * invOfUnit φ u = 1 :=
  MvPowerSeries.mul_invOfUnit φ u <| h


@[simp]
theorem invOfUnit_mul (φ : R⟦X⟧) (u : Rˣ) (h : constantCoeff R φ = u) :
    invOfUnit φ u * φ = 1 :=
  MvPowerSeries.invOfUnit_mul φ u h


theorem isUnit_iff_constantCoeff {φ : R⟦X⟧} :
    IsUnit φ ↔ IsUnit (constantCoeff R φ) :=
  MvPowerSeries.isUnit_iff_constantCoeff


/-- Two ways of removing the constant coefficient of a power series are the same. -/
theorem sub_const_eq_shift_mul_X (φ : R⟦X⟧) :
    φ - C R (constantCoeff R φ) = (mk fun p ↦ coeff R (p + 1) φ) * X :=
  sub_eq_iff_eq_add.mpr (eq_shift_mul_X_add_const φ)


theorem sub_const_eq_X_mul_shift (φ : R⟦X⟧) :
    φ - C R (constantCoeff R φ) = X * mk fun p ↦ coeff R (p + 1) φ :=
  sub_eq_iff_eq_add.mpr (eq_X_mul_shift_add_const φ)


/-- The inverse 1/f of a power series f defined over a field -/
protected def inv : k⟦X⟧ → k⟦X⟧ :=
  MvPowerSeries.inv


instance : Inv k⟦X⟧ := ⟨PowerSeries.inv⟩


theorem inv_eq_inv_aux (φ : k⟦X⟧) : φ⁻¹ = inv.aux (constantCoeff k φ)⁻¹ φ :=
  rfl


theorem coeff_inv (n) (φ : k⟦X⟧) :
    coeff k n φ⁻¹ =
      if n = 0 then (constantCoeff k φ)⁻¹
      else
        -(constantCoeff k φ)⁻¹ *
          ∑ x ∈ antidiagonal n,
            if x.2 < n then coeff k x.1 φ * coeff k x.2 φ⁻¹ else 0 := by
  /-
    k : Type u_2
    inst✝ : Field k
    n : Nat
    φ : PowerSeries k
    ⊢ Eq ((PowerSeries.coeff k n) (Inv.inv φ)) (ite (Eq n 0) (Inv.inv ((PowerSerie …
  -/
  rw [inv_eq_inv_aux, coeff_inv_aux n (constantCoeff k φ)⁻¹ φ]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_inv (φ : k⟦X⟧) : constantCoeff k φ⁻¹ = (constantCoeff k φ)⁻¹ :=
  MvPowerSeries.constantCoeff_inv φ


theorem inv_eq_zero {φ : k⟦X⟧} : φ⁻¹ = 0 ↔ constantCoeff k φ = 0 :=
  MvPowerSeries.inv_eq_zero


theorem zero_inv : (0 : k⟦X⟧)⁻¹ = 0 :=
  MvPowerSeries.zero_inv


@[simp]
theorem invOfUnit_eq (φ : k⟦X⟧) (h : constantCoeff k φ ≠ 0) :
    invOfUnit φ (Units.mk0 _ h) = φ⁻¹ :=
  MvPowerSeries.invOfUnit_eq _ _


@[simp]
theorem invOfUnit_eq' (φ : k⟦X⟧) (u : Units k) (h : constantCoeff k φ = u) :
    invOfUnit φ u = φ⁻¹ :=
  MvPowerSeries.invOfUnit_eq' φ _ h


@[simp]
protected theorem mul_inv_cancel (φ : k⟦X⟧) (h : constantCoeff k φ ≠ 0) : φ * φ⁻¹ = 1 :=
  MvPowerSeries.mul_inv_cancel φ h


@[simp]
protected theorem inv_mul_cancel (φ : k⟦X⟧) (h : constantCoeff k φ ≠ 0) : φ⁻¹ * φ = 1 :=
  MvPowerSeries.inv_mul_cancel φ h


theorem eq_mul_inv_iff_mul_eq {φ₁ φ₂ φ₃ : k⟦X⟧} (h : constantCoeff k φ₃ ≠ 0) :
    φ₁ = φ₂ * φ₃⁻¹ ↔ φ₁ * φ₃ = φ₂ :=
  MvPowerSeries.eq_mul_inv_iff_mul_eq h


theorem eq_inv_iff_mul_eq_one {φ ψ : k⟦X⟧} (h : constantCoeff k ψ ≠ 0) :
    φ = ψ⁻¹ ↔ φ * ψ = 1 :=
  MvPowerSeries.eq_inv_iff_mul_eq_one h


theorem inv_eq_iff_mul_eq_one {φ ψ : k⟦X⟧} (h : constantCoeff k ψ ≠ 0) :
    ψ⁻¹ = φ ↔ φ * ψ = 1 :=
  MvPowerSeries.inv_eq_iff_mul_eq_one h


protected theorem mul_inv_rev (φ ψ : k⟦X⟧) : (φ * ψ)⁻¹ = ψ⁻¹ * φ⁻¹ :=
  MvPowerSeries.mul_inv_rev _ _


instance : InvOneClass k⟦X⟧ :=
  { inferInstanceAs <| InvOneClass <| MvPowerSeries Unit k with }


@[simp]
theorem C_inv (r : k) : (C k r)⁻¹ = C k r⁻¹ :=
  MvPowerSeries.C_inv _


@[simp]
theorem X_inv : (X : k⟦X⟧)⁻¹ = 0 :=
  MvPowerSeries.X_inv _


theorem smul_inv (r : k) (φ : k⟦X⟧) : (r • φ)⁻¹ = r⁻¹ • φ⁻¹ :=
  MvPowerSeries.smul_inv _ _


/-- `firstUnitCoeff` is the non-zero coefficient whose index is `f.order`, seen as a unit of the
  field. It is obtained using `divided_by_X_pow_order`, defined in `PowerSeries.Order`-/
def firstUnitCoeff {f : k⟦X⟧} (hf : f ≠ 0) : kˣ :=
  let d := f.order.lift (order_finite_iff_ne_zero.mpr hf)
                                       /-
                                         R : Type u_1
                                         k : Type u_2
                                         inst✝ : Field k
                                         f : PowerSeries k
                                         hf : Ne f 0
                                         d : Nat := f.order.lift ⋯
                                         ⊢ Ne ((PowerSeries.coeff k d) f) 0
                                       -/
  have f_const : coeff k d f ≠ 0 := by apply coeff_order
                                       /-
                                         🎉 no goals
                                       -/
  have : Invertible (constantCoeff k (divided_by_X_pow_order hf)) := by
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      d : Nat := f.order.lift ⋯
      f_const : Ne ((PowerSeries.coeff k d) f) 0
      ⊢ Invertible ((PowerSeries.constantCoeff k) (PowerSeries.divided_by_X_pow_orde …
    -/
    apply invertibleOfNonzero
    /-
      case h
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      d : Nat := f.order.lift ⋯
      f_const : Ne ((PowerSeries.coeff k d) f) 0
      ⊢ Ne ((PowerSeries.constantCoeff k) (PowerSeries.divided_by_X_pow_order hf)) 0
    -/
    convert f_const using 1
    /-
      case h.e'_2
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      d : Nat := f.order.lift ⋯
      f_const : Ne ((PowerSeries.coeff k d) f) 0
      ⊢ Eq ((PowerSeries.constantCoeff k) (PowerSeries.divided_by_X_pow_order hf)) ( …
    -/
    rw [← coeff_zero_eq_constantCoeff, ← zero_add d]
    convert (coeff_X_pow_mul (exists_eq_mul_right_of_dvd (X_pow_order_dvd
      (order_finite_iff_ne_zero.mpr hf))).choose d 0).symm
    /-
      case h.e'_3.h.e'_6
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      d : Nat := f.order.lift ⋯
      f_const : Ne ((PowerSeries.coeff k d) f) 0
      ⊢ Eq f (HMul.hMul (HPow.hPow PowerSeries.X d) ⋯.choose)
    -/
    exact (self_eq_X_pow_order_mul_divided_by_X_pow_order hf).symm
    /-
      🎉 no goals
    -/
  unitOfInvertible (constantCoeff k (divided_by_X_pow_order hf))


/-- `Inv_divided_by_X_pow_order` is the inverse of the element obtained by diving a non-zero power
series by the largest power of `X` dividing it. Useful to create a term of type `Units`, done in
`Unit_divided_by_X_pow_order` -/
def Inv_divided_by_X_pow_order {f : k⟦X⟧} (hf : f ≠ 0) : k⟦X⟧ :=
  invOfUnit (divided_by_X_pow_order hf) (firstUnitCoeff hf)


@[simp]
theorem Inv_divided_by_X_pow_order_rightInv {f : k⟦X⟧} (hf : f ≠ 0) :
    divided_by_X_pow_order hf * Inv_divided_by_X_pow_order hf = 1 :=
  mul_invOfUnit (divided_by_X_pow_order hf) (firstUnitCoeff hf) rfl


@[simp]
theorem Inv_divided_by_X_pow_order_leftInv {f : k⟦X⟧} (hf : f ≠ 0) :
    (Inv_divided_by_X_pow_order hf) * (divided_by_X_pow_order hf) = 1 := by
  /-
    k : Type u_2
    inst✝ : Field k
    f : PowerSeries k
    hf : Ne f 0
    ⊢ Eq (HMul.hMul (PowerSeries.Inv_divided_by_X_pow_order hf) (PowerSeries.divid …
  -/
  rw [mul_comm]
  /-
    k : Type u_2
    inst✝ : Field k
    f : PowerSeries k
    hf : Ne f 0
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.Inv_divid …
  -/
  exact mul_invOfUnit (divided_by_X_pow_order hf) (firstUnitCoeff hf) rfl
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- `Unit_of_divided_by_X_pow_order` is the unit power series obtained by dividing a non-zero
power series by the largest power of `X` that divides it. -/
def Unit_of_divided_by_X_pow_order (f : k⟦X⟧) : k⟦X⟧ˣ :=
  if hf : f = 0 then 1
  else
    { val := divided_by_X_pow_order hf
      inv := Inv_divided_by_X_pow_order hf
      val_inv := Inv_divided_by_X_pow_order_rightInv hf
      inv_val := Inv_divided_by_X_pow_order_leftInv hf }


theorem isUnit_divided_by_X_pow_order {f : k⟦X⟧} (hf : f ≠ 0) :
    IsUnit (divided_by_X_pow_order hf) :=
  ⟨Unit_of_divided_by_X_pow_order f,
       /-
         k : Type u_2
         inst✝ : Field k
         f : PowerSeries k
         hf : Ne f 0
         ⊢ Eq (↑f.Unit_of_divided_by_X_pow_order) (PowerSeries.divided_by_X_pow_order hf)
       -/
    by simp only [Unit_of_divided_by_X_pow_order, dif_neg hf, Units.val_mk]⟩
       /-
         🎉 no goals
       -/


theorem Unit_of_divided_by_X_pow_order_nonzero {f : k⟦X⟧} (hf : f ≠ 0) :
    ↑(Unit_of_divided_by_X_pow_order f) = divided_by_X_pow_order hf := by
  /-
    k : Type u_2
    inst✝ : Field k
    f : PowerSeries k
    hf : Ne f 0
    ⊢ Eq (↑f.Unit_of_divided_by_X_pow_order) (PowerSeries.divided_by_X_pow_order hf)
  -/
  simp only [Unit_of_divided_by_X_pow_order, dif_neg hf, Units.val_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem Unit_of_divided_by_X_pow_order_zero : Unit_of_divided_by_X_pow_order (0 : k⟦X⟧) = 1 := by
  /-
    k : Type u_2
    inst✝ : Field k
    ⊢ Eq (PowerSeries.Unit_of_divided_by_X_pow_order 0) 1
  -/
  simp only [Unit_of_divided_by_X_pow_order, dif_pos]
  /-
    🎉 no goals
  -/


theorem eq_divided_by_X_pow_order_Iff_Unit {f : k⟦X⟧} (hf : f ≠ 0) :
    f = divided_by_X_pow_order hf ↔ IsUnit f :=
              /-
                k : Type u_2
                inst✝ : Field k
                f : PowerSeries k
                hf : Ne f 0
                h : Eq f (PowerSeries.divided_by_X_pow_order hf)
                ⊢ IsUnit f
              -/
  ⟨fun h ↦ by rw [h]; exact isUnit_divided_by_X_pow_order hf, fun h ↦ by
                      /-
                        🎉 no goals
                      -/
    have : f.order.lift (order_finite_iff_ne_zero.mpr hf) = 0 := by
      simp [order_zero_of_unit h]
    /-
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      h : IsUnit f
      this : Eq (f.order.lift ⋯) 0
      ⊢ Eq f (PowerSeries.divided_by_X_pow_order hf)
    -/
    convert (self_eq_X_pow_order_mul_divided_by_X_pow_order hf).symm
    /-
      case h.e'_3
      k : Type u_2
      inst✝ : Field k
      f : PowerSeries k
      hf : Ne f 0
      h : IsUnit f
      this : Eq (f.order.lift ⋯) 0
      ⊢ Eq (PowerSeries.divided_by_X_pow_order hf) (HMul.hMul (HPow.hPow PowerSeries …
    -/
    simp only [this, pow_zero, one_mul]⟩
    /-
      🎉 no goals
    -/


@[instance]
theorem map.isLocalHom : IsLocalHom (map f) :=
  MvPowerSeries.map.isLocalHom f


@[deprecated (since := "2024-10-10")]
alias map.isLocalRingHom := map.isLocalHom


instance : IsLocalRing R⟦X⟧ :=
  { inferInstanceAs <| IsLocalRing <| MvPowerSeries Unit R with }



theorem hasUnitMulPowIrreducibleFactorization :
    HasUnitMulPowIrreducibleFactorization k⟦X⟧ :=
  ⟨X, And.intro X_irreducible
      (by
        /-
          k : Type u_2
          inst✝ : Field k
          ⊢ ∀ {x : PowerSeries k}, Ne x 0 → Exists fun n => Associated (HPow.hPow PowerS …
        -/
        intro f hf
        /-
          k : Type u_2
          inst✝ : Field k
          f : PowerSeries k
          hf : Ne f 0
          ⊢ Exists fun n => Associated (HPow.hPow PowerSeries.X n) f
        -/
        use f.order.lift (order_finite_iff_ne_zero.mpr hf)
        /-
          case h
          k : Type u_2
          inst✝ : Field k
          f : PowerSeries k
          hf : Ne f 0
          ⊢ Associated (HPow.hPow PowerSeries.X (f.order.lift ⋯)) f
        -/
        use Unit_of_divided_by_X_pow_order f
        /-
          case h
          k : Type u_2
          inst✝ : Field k
          f : PowerSeries k
          hf : Ne f 0
          ⊢ Eq (HMul.hMul (HPow.hPow PowerSeries.X (f.order.lift ⋯)) ↑f.Unit_of_divided_ …
        -/
        simp only [Unit_of_divided_by_X_pow_order_nonzero hf]
        /-
          case h
          k : Type u_2
          inst✝ : Field k
          f : PowerSeries k
          hf : Ne f 0
          ⊢ Eq (HMul.hMul (HPow.hPow PowerSeries.X (f.order.lift ⋯)) (PowerSeries.divide …
        -/
        exact self_eq_X_pow_order_mul_divided_by_X_pow_order hf)⟩
        /-
          🎉 no goals
        -/


instance : UniqueFactorizationMonoid k⟦X⟧ :=
  hasUnitMulPowIrreducibleFactorization.toUniqueFactorizationMonoid


instance : IsDiscreteValuationRing k⟦X⟧ :=
  ofHasUnitMulPowIrreducibleFactorization hasUnitMulPowIrreducibleFactorization


instance isNoetherianRing : IsNoetherianRing k⟦X⟧ :=
  PrincipalIdealRing.isNoetherianRing


/-- The maximal ideal of `k⟦X⟧` is generated by `X`. -/
theorem maximalIdeal_eq_span_X : IsLocalRing.maximalIdeal (k⟦X⟧) = Ideal.span {X} := by
  have hX : (Ideal.span {(X : k⟦X⟧)}).IsMaximal := by
    rw [Ideal.isMaximal_iff]
    constructor
    · rw [Ideal.mem_span_singleton]
      exact Prime.not_dvd_one X_prime
    · intro I f hI hfX hfI
      rw [Ideal.mem_span_singleton, X_dvd_iff] at hfX
      have hfI0 : C k (f 0) ∈ I := by
        have : C k (f 0) = f - (f - C k (f 0)) := by rw [sub_sub_cancel]
        rw [this]
        apply Ideal.sub_mem I hfI
        apply hI
        rw [Ideal.mem_span_singleton, X_dvd_iff, map_sub, constantCoeff_C, ←
          coeff_zero_eq_constantCoeff_apply, sub_eq_zero, coeff_zero_eq_constantCoeff]
        rfl
      rw [← Ideal.eq_top_iff_one]
      apply Ideal.eq_top_of_isUnit_mem I hfI0 (IsUnit.map (C k) (Ne.isUnit hfX))
  /-
    k : Type u_2
    inst✝ : Field k
    hX : (Ideal.span (Singleton.singleton PowerSeries.X)).IsMaximal
    ⊢ Eq (IsLocalRing.maximalIdeal (PowerSeries k)) (Ideal.span (Singleton.singlet …
  -/
  rw [IsLocalRing.eq_maximalIdeal hX]
  /-
    🎉 no goals
  -/


instance : NormalizationMonoid k⟦X⟧ where
  normUnit f := (Unit_of_divided_by_X_pow_order f)⁻¹
                      /-
                        R : Type u_1
                        k : Type u_2
                        inst✝ : Field k
                        ⊢ Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_pow_order) 0) 1
                      -/
  normUnit_zero := by simp only [Unit_of_divided_by_X_pow_order_zero, inv_one]
                      /-
                        🎉 no goals
                      -/
  normUnit_mul  := fun hf hg ↦ by
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      a✝ b✝ : PowerSeries k
      hf : Ne a✝ 0
      hg : Ne b✝ 0
      ⊢ Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_pow_order) (HMul.hMul a✝ b✝)) ( …
    -/
    simp only [← mul_inv, inv_inj]
    simp only [Unit_of_divided_by_X_pow_order_nonzero (mul_ne_zero hf hg),
      Unit_of_divided_by_X_pow_order_nonzero hf, Unit_of_divided_by_X_pow_order_nonzero hg,
      Units.ext_iff, val_unitOfInvertible, Units.val_mul, divided_by_X_pow_orderMul]
  normUnit_coe_units := by
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      ⊢ ∀ (u : Units (PowerSeries k)), Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_ …
    -/
    intro u
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      u : Units (PowerSeries k)
      ⊢ Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_pow_order) ↑u) (Inv.inv u)
    -/
    set u₀ := u.1 with hu
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      u : Units (PowerSeries k)
      u₀ : PowerSeries k := ↑u
      hu : Eq u₀ ↑u
      ⊢ Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_pow_order) u₀) (Inv.inv u)
    -/
    have h₀ : IsUnit u₀ := ⟨u, hu.symm⟩
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      u : Units (PowerSeries k)
      u₀ : PowerSeries k := ↑u
      hu : Eq u₀ ↑u
      h₀ : IsUnit u₀
      ⊢ Eq ((fun f => Inv.inv f.Unit_of_divided_by_X_pow_order) u₀) (Inv.inv u)
    -/
    rw [inv_inj, Units.ext_iff, ← hu, Unit_of_divided_by_X_pow_order_nonzero h₀.ne_zero]
    /-
      R : Type u_1
      k : Type u_2
      inst✝ : Field k
      u : Units (PowerSeries k)
      u₀ : PowerSeries k := ↑u
      hu : Eq u₀ ↑u
      h₀ : IsUnit u₀
      ⊢ Eq (PowerSeries.divided_by_X_pow_order ⋯) u₀
    -/
    exact ((eq_divided_by_X_pow_order_Iff_Unit h₀.ne_zero).mpr h₀).symm
    /-
      🎉 no goals
    -/


theorem normUnit_X : normUnit (X : k⟦X⟧) = 1 := by
  /-
    k : Type u_2
    inst✝ : Field k
    ⊢ Eq (NormalizationMonoid.normUnit PowerSeries.X) 1
  -/
  simp [normUnit, ← Units.val_eq_one, Unit_of_divided_by_X_pow_order_nonzero]
  /-
    🎉 no goals
  -/


theorem X_eq_normalizeX : (X : k⟦X⟧) = normalize X := by
  /-
    k : Type u_2
    inst✝ : Field k
    ⊢ Eq PowerSeries.X (normalize PowerSeries.X)
  -/
  simp only [normalize_apply, normUnit_X, Units.val_one, mul_one]
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem normalized_count_X_eq_of_coe {P : k[X]} (hP : P ≠ 0) :
    Multiset.count PowerSeries.X (normalizedFactors (P : k⟦X⟧)) =
      Multiset.count Polynomial.X (normalizedFactors P) := by
  /-
    k : Type u_2
    inst✝ : Field k
    P : Polynomial k
    hP : Ne P 0
    ⊢ Eq (Multiset.count PowerSeries.X (UniqueFactorizationMonoid.normalizedFactor …
  -/
  apply eq_of_forall_le_iff
  /-
    case H
    k : Type u_2
    inst✝ : Field k
    P : Polynomial k
    hP : Ne P 0
    ⊢ ∀ (c : Nat), Iff (LE.le c (Multiset.count PowerSeries.X (UniqueFactorization …
  -/
  simp only [← Nat.cast_le (α := ℕ∞)]
  rw [X_eq_normalize, PowerSeries.X_eq_normalizeX, ← emultiplicity_eq_count_normalizedFactors
    irreducible_X hP, ← emultiplicity_eq_count_normalizedFactors X_irreducible] <;>
  simp only [← pow_dvd_iff_le_emultiplicity, Polynomial.X_pow_dvd_iff,
    PowerSeries.X_pow_dvd_iff, Polynomial.coeff_coe P, implies_true, ne_eq, coe_eq_zero_iff, hP,
    not_false_eq_true]


theorem ker_coeff_eq_max_ideal : RingHom.ker (constantCoeff k) = maximalIdeal _ :=
  Ideal.ext fun _ ↦ by
    /-
      k : Type u_2
      inst✝ : Field k
      x✝ : PowerSeries k
      ⊢ Iff (Membership.mem (RingHom.ker (PowerSeries.constantCoeff k)) x✝) (Members …
    -/
    rw [RingHom.mem_ker, maximalIdeal_eq_span_X, Ideal.mem_span_singleton, X_dvd_iff]
    /-
      🎉 no goals
    -/


/-- The ring isomorphism between the residue field of the ring of power series valued in a field `K`
and `K` itself. -/
def residueFieldOfPowerSeries : ResidueField k⟦X⟧ ≃+* k :=
  (Ideal.quotEquivOfEq (ker_coeff_eq_max_ideal).symm).trans
    (RingHom.quotientKerEquivOfSurjective constantCoeff_surj)


