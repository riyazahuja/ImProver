/-- The power series for `1 / (u - x)`. -/
def invUnitsSub (u : Rˣ) : PowerSeries R :=
  mk fun n => 1 /ₚ u ^ (n + 1)


@[simp]
theorem coeff_invUnitsSub (u : Rˣ) (n : ℕ) : coeff R n (invUnitsSub u) = 1 /ₚ u ^ (n + 1) :=
  coeff_mk _ _


@[simp]
theorem constantCoeff_invUnitsSub (u : Rˣ) : constantCoeff R (invUnitsSub u) = 1 /ₚ u := by
  /-
    R : Type u_1
    inst✝ : Ring R
    u : Units R
    ⊢ Eq ((PowerSeries.constantCoeff R) (PowerSeries.invUnitsSub u)) (divp 1 u)
  -/
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_invUnitsSub, zero_add, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem invUnitsSub_mul_X (u : Rˣ) : invUnitsSub u * X = invUnitsSub u * C R u - 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    u : Units R
    ⊢ Eq (HMul.hMul (PowerSeries.invUnitsSub u) PowerSeries.X) (HSub.hSub (HMul.hM …
  -/
  ext (_ | n)
    /-
      case h.zero
      R : Type u_1
      inst✝ : Ring R
      u : Units R
      ⊢ Eq ((PowerSeries.coeff R 0) (HMul.hMul (PowerSeries.invUnitsSub u) PowerSeri …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      R : Type u_1
      inst✝ : Ring R
      u : Units R
      n : Nat
      ⊢ Eq ((PowerSeries.coeff R (HAdd.hAdd n 1)) (HMul.hMul (PowerSeries.invUnitsSu …
    -/
  · simp [n.succ_ne_zero, pow_succ']
    /-
      🎉 no goals
    -/


@[simp]
theorem invUnitsSub_mul_sub (u : Rˣ) : invUnitsSub u * (C R u - X) = 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    u : Units R
    ⊢ Eq (HMul.hMul (PowerSeries.invUnitsSub u) (HSub.hSub ((PowerSeries.C R) ↑u)  …
  -/
  simp [mul_sub, sub_sub_cancel]
  /-
    🎉 no goals
  -/


theorem map_invUnitsSub (f : R →+* S) (u : Rˣ) :
    map f (invUnitsSub u) = invUnitsSub (Units.map (f : R →* S) u) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    u : Units R
    ⊢ Eq ((PowerSeries.map f) (PowerSeries.invUnitsSub u)) (PowerSeries.invUnitsSu …
  -/
  ext
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    u : Units R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff S n✝) ((PowerSeries.map f) (PowerSeries.invUnitsSub u …
  -/
  simp only [← map_pow, coeff_map, coeff_invUnitsSub, one_divp]
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    u : Units R
    n✝ : Nat
    ⊢ Eq (f ↑(Inv.inv (HPow.hPow u (HAdd.hAdd n✝ 1)))) ↑(Inv.inv ((Units.map ↑f) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
(1 + X + X^2 + ...) * (1 - X) = 1.

Note that the power series `1 + X + X^2 + ...` is written as `mk 1` where `1` is the constant
function so that `mk 1` is the power series with all coefficients equal to one.
-/
theorem mk_one_mul_one_sub_eq_one : (mk 1 : S⟦X⟧) * (1 - X) = 1 := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (HMul.hMul (PowerSeries.mk 1) (HSub.hSub 1 PowerSeries.X)) 1
  -/
  rw [mul_comm, PowerSeries.ext_iff]
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ ∀ (n : Nat), Eq ((PowerSeries.coeff S n) (HMul.hMul (HSub.hSub 1 PowerSeries …
  -/
  intro n
  cases n with
  | zero => simp
  | succ n => simp [sub_mul]


/--
Note that `mk 1` is the constant function `1` so the power series `1 + X + X^2 + ...`. This theorem
states that for any `d : ℕ`, `(1 + X + X^2 + ... : S⟦X⟧) ^ (d + 1)` is equal to the power series
`mk fun n => Nat.choose (d + n) d : S⟦X⟧`.
-/
theorem mk_one_pow_eq_mk_choose_add :
    (mk 1 : S⟦X⟧) ^ (d + 1) = (mk fun n => Nat.choose (d + n) d : S⟦X⟧) := by
  induction d with
  | zero => ext; simp
  | succ d hd =>
      ext n
      rw [pow_add, hd, pow_one, mul_comm, coeff_mul]
      simp_rw [coeff_mk, Pi.one_apply, one_mul]
      norm_cast
      rw [Finset.sum_antidiagonal_choose_add, ← Nat.choose_succ_succ, Nat.succ_eq_add_one,
        add_right_comm]


/--
Given a natural number `d : ℕ` and a commutative ring `S`, `PowerSeries.invOneSubPow S d` is the
multiplicative inverse of `(1 - X) ^ d` in `S⟦X⟧ˣ`. When `d` is `0`, `PowerSeries.invOneSubPow S d`
will just be `1`. When `d` is positive, `PowerSeries.invOneSubPow S d` will be the power series
`mk fun n => Nat.choose (d - 1 + n) (d - 1)`.
-/
noncomputable def invOneSubPow : ℕ → S⟦X⟧ˣ
  | 0 => 1
  | d + 1 => {
    val := mk fun n => Nat.choose (d + n) d
    inv := (1 - X) ^ (d + 1)
    val_inv := by
      /-
        S : Type u_1
        inst✝ : CommRing S
        d✝ d : Nat
        ⊢ Eq (HMul.hMul (PowerSeries.mk fun n => ↑((HAdd.hAdd d n).choose d)) (HPow.hP …
      -/
      rw [← mk_one_pow_eq_mk_choose_add, ← mul_pow, mk_one_mul_one_sub_eq_one, one_pow]
      /-
        🎉 no goals
      -/
    inv_val := by
      /-
        S : Type u_1
        inst✝ : CommRing S
        d✝ d : Nat
        ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 PowerSeries.X) (HAdd.hAdd d 1)) (Power …
      -/
      rw [← mk_one_pow_eq_mk_choose_add, ← mul_pow, mul_comm, mk_one_mul_one_sub_eq_one, one_pow]
      /-
        🎉 no goals
      -/
    }


theorem invOneSubPow_zero : invOneSubPow S 0 = 1 := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (PowerSeries.invOneSubPow S 0) 1
  -/
  delta invOneSubPow
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (PowerSeries.invOneSubPow.match_1 (fun x => Units (PowerSeries S)) 0 (fun …
  -/
  simp only [Units.val_one]
  /-
    🎉 no goals
  -/


theorem invOneSubPow_val_eq_mk_sub_one_add_choose_of_pos (h : 0 < d) :
    (invOneSubPow S d).val = (mk fun n => Nat.choose (d - 1 + n) (d - 1) : S⟦X⟧) := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    d : Nat
    h : LT.lt 0 d
    ⊢ Eq (↑(PowerSeries.invOneSubPow S d)) (PowerSeries.mk fun n => ↑((HAdd.hAdd ( …
  -/
  rw [← Nat.sub_one_add_one_eq_of_pos h, invOneSubPow, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


theorem invOneSubPow_val_succ_eq_mk_add_choose :
    (invOneSubPow S (d + 1)).val = (mk fun n => Nat.choose (d + n) d : S⟦X⟧) := rfl


theorem invOneSubPow_val_one_eq_invUnitSub_one :
    (invOneSubPow S 1).val = invUnitsSub (1 : Sˣ) := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (↑(PowerSeries.invOneSubPow S 1)) (PowerSeries.invUnitsSub 1)
  -/
  simp [invOneSubPow, invUnitsSub]
  /-
    🎉 no goals
  -/


/--
The theorem `PowerSeries.mk_one_mul_one_sub_eq_one` implies that `1 - X` is a unit in `S⟦X⟧`
whose inverse is the power series `1 + X + X^2 + ...`. This theorem states that for any `d : ℕ`,
`PowerSeries.invOneSubPow S d` is equal to `(1 - X)⁻¹ ^ d`.
-/
theorem invOneSubPow_eq_inv_one_sub_pow :
    invOneSubPow S d =
      (Units.mkOfMulEqOne (1 - X) (mk 1 : S⟦X⟧) <|
        Eq.trans (mul_comm _ _) (mk_one_mul_one_sub_eq_one S))⁻¹ ^ d := by
  induction d with
  | zero => exact Eq.symm <| pow_zero _
  | succ d _ =>
      rw [inv_pow]
      exact (DivisionMonoid.inv_eq_of_mul _ (invOneSubPow S (d + 1)) <| by
        rw [← Units.val_eq_one, Units.val_mul, Units.val_pow_eq_pow_val]
        exact (invOneSubPow S (d + 1)).inv_val).symm


theorem invOneSubPow_inv_eq_one_sub_pow :
    (invOneSubPow S d).inv = (1 - X : S⟦X⟧) ^ d := by
  induction d with
  | zero => exact Eq.symm <| pow_zero _
  | succ d => rfl


theorem invOneSubPow_inv_zero_eq_one : (invOneSubPow S 0).inv = 1 := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (PowerSeries.invOneSubPow S 0).inv 1
  -/
  delta invOneSubPow
  /-
    S : Type u_1
    inst✝ : CommRing S
    ⊢ Eq (PowerSeries.invOneSubPow.match_1 (fun x => Units (PowerSeries S)) 0 (fun …
  -/
  simp only [Units.inv_eq_val_inv, inv_one, Units.val_one]
  /-
    🎉 no goals
  -/


theorem mk_add_choose_mul_one_sub_pow_eq_one :
    (mk fun n ↦ Nat.choose (d + n) d : S⟦X⟧) * ((1 - X) ^ (d + 1)) = 1 :=
  (invOneSubPow S (d + 1)).val_inv


theorem invOneSubPow_add (e : ℕ) :
    invOneSubPow S (d + e) = invOneSubPow S d * invOneSubPow S e := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    d e : Nat
    ⊢ Eq (PowerSeries.invOneSubPow S (HAdd.hAdd d e)) (HMul.hMul (PowerSeries.invO …
  -/
  simp_rw [invOneSubPow_eq_inv_one_sub_pow, pow_add]
  /-
    🎉 no goals
  -/


theorem one_sub_pow_mul_invOneSubPow_val_add_eq_invOneSubPow_val (e : ℕ) :
    (1 - X) ^ e * (invOneSubPow S (d + e)).val = (invOneSubPow S d).val := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    d e : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 PowerSeries.X) e) ↑(PowerSeries.invOne …
  -/
  simp [invOneSubPow_add, Units.val_mul, mul_comm, mul_assoc, ← invOneSubPow_inv_eq_one_sub_pow]
  /-
    🎉 no goals
  -/


theorem one_sub_pow_add_mul_invOneSubPow_val_eq_one_sub_pow (e : ℕ) :
    (1 - X) ^ (d + e) * (invOneSubPow S e).val = (1 - X) ^ d := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    d e : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 PowerSeries.X) (HAdd.hAdd d e)) ↑(Powe …
  -/
  simp [pow_add, mul_assoc, ← invOneSubPow_inv_eq_one_sub_pow S e]
  /-
    🎉 no goals
  -/


/-- Power series for the exponential function at zero. -/
def exp : PowerSeries A :=
  mk fun n => algebraMap ℚ A (1 / n !)


/-- Power series for the sine function at zero. -/
def sin : PowerSeries A :=
  mk fun n => if Even n then 0 else algebraMap ℚ A ((-1) ^ (n / 2) / n !)


/-- Power series for the cosine function at zero. -/
def cos : PowerSeries A :=
  mk fun n => if Even n then algebraMap ℚ A ((-1) ^ (n / 2) / n !) else 0


@[simp]
theorem coeff_exp : coeff A n (exp A) = algebraMap ℚ A (1 / n !) :=
  coeff_mk _ _


@[simp]
theorem constantCoeff_exp : constantCoeff A (exp A) = 1 := by
  /-
    A : Type u_1
    inst✝¹ : Ring A
    inst✝ : Algebra Rat A
    ⊢ Eq ((PowerSeries.constantCoeff A) (PowerSeries.exp A)) 1
  -/
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_exp]
  /-
    A : Type u_1
    inst✝¹ : Ring A
    inst✝ : Algebra Rat A
    ⊢ Eq ((algebraMap Rat A) (HDiv.hDiv 1 ↑(Nat.factorial 0))) 1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_exp : map (f : A →+* A') (exp A) = exp A' := by
  /-
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    ⊢ Eq ((PowerSeries.map f) (PowerSeries.exp A)) (PowerSeries.exp A')
  -/
  ext
  /-
    case h
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff A' n✝) ((PowerSeries.map f) (PowerSeries.exp A))) ((P …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_sin : map f (sin A) = sin A' := by
  /-
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    ⊢ Eq ((PowerSeries.map f) (PowerSeries.sin A)) (PowerSeries.sin A')
  -/
  ext
  /-
    case h
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff A' n✝) ((PowerSeries.map f) (PowerSeries.sin A))) ((P …
  -/
  simp [sin, apply_ite f]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_cos : map f (cos A) = cos A' := by
  /-
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    ⊢ Eq ((PowerSeries.map f) (PowerSeries.cos A)) (PowerSeries.cos A')
  -/
  ext
  /-
    case h
    A : Type u_1
    A' : Type u_2
    inst✝³ : Ring A
    inst✝² : Ring A'
    inst✝¹ : Algebra Rat A
    inst✝ : Algebra Rat A'
    f : RingHom A A'
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff A' n✝) ((PowerSeries.map f) (PowerSeries.cos A))) ((P …
  -/
  simp [cos, apply_ite f]
  /-
    🎉 no goals
  -/


/-- Shows that $e^{aX} * e^{bX} = e^{(a + b)X}$ -/
theorem exp_mul_exp_eq_exp_add [Algebra ℚ A] (a b : A) :
    rescale a (exp A) * rescale b (exp A) = rescale (a + b) (exp A) := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    ⊢ Eq (HMul.hMul ((PowerSeries.rescale a) (PowerSeries.exp A)) ((PowerSeries.re …
  -/
  ext n
  simp only [coeff_mul, exp, rescale, coeff_mk, MonoidHom.coe_mk, OneHom.coe_mk, coe_mk,
    factorial, Nat.sum_antidiagonal_eq_sum_range_succ_mk, add_pow, sum_mul]
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n : Nat
    ⊢ Eq ((Finset.range n.succ).sum fun x => HMul.hMul (HMul.hMul (HPow.hPow a x)  …
  -/
  apply sum_congr rfl
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n.succ) x → Eq (HMul.hMul (HMul.hM …
  -/
  rintro x hx
  suffices
    a ^ x * b ^ (n - x) *
        (algebraMap ℚ A (1 / ↑x.factorial) * algebraMap ℚ A (1 / ↑(n - x).factorial)) =
      a ^ x * b ^ (n - x) * (↑(n.choose x) * (algebraMap ℚ A) (1 / ↑n.factorial))
    by convert this using 1 <;> ring
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a x) (HPow.hPow b (HSub.hSub n x))) (HMu …
  -/
  congr 1
  /-
    case h.e_a
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv 1 ↑x.factorial)) ((algebraMap R …
  -/
  rw [← map_natCast (algebraMap ℚ A) (n.choose x), ← map_mul, ← map_mul]
  /-
    case h.e_a
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq ((algebraMap Rat A) (HMul.hMul (HDiv.hDiv 1 ↑x.factorial) (HDiv.hDiv 1 ↑( …
  -/
  refine RingHom.congr_arg _ ?_
  /-
    case h.e_a
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑x.factorial) (HDiv.hDiv 1 ↑(HSub.hSub n x).facto …
  -/
  rw [mul_one_div (↑(n.choose x) : ℚ), one_div_mul_one_div]
  /-
    case h.e_a
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HDiv.hDiv 1 (HMul.hMul ↑x.factorial ↑(HSub.hSub n x).factorial)) (HDiv.h …
  -/
  symm
  /-
    case h.e_a
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    a b : A
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HDiv.hDiv ↑(n.choose x) ↑n.factorial) (HDiv.hDiv 1 (HMul.hMul ↑x.factori …
  -/
  rw [div_eq_iff, div_mul_eq_mul_div, one_mul, choose_eq_factorial_div_factorial]
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      ⊢ Eq (↑(HDiv.hDiv n.factorial (HMul.hMul x.factorial (HSub.hSub n x).factorial …
    -/
  · norm_cast
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      ⊢ Eq (↑(HDiv.hDiv n.factorial (HMul.hMul x.factorial (HSub.hSub n x).factorial …
    -/
    rw [cast_div_charZero]
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      ⊢ Dvd.dvd (HMul.hMul x.factorial (HSub.hSub n x).factorial) n.factorial
    -/
    apply factorial_mul_factorial_dvd_factorial (mem_range_succ_iff.1 hx)
    /-
      🎉 no goals
    -/
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      ⊢ LE.le x n
    -/
  · apply mem_range_succ_iff.1 hx
    /-
      🎉 no goals
    -/
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      ⊢ Ne (↑n.factorial) 0
    -/
  · rintro h
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      h : Eq (↑n.factorial) 0
      ⊢ False
    -/
    apply factorial_ne_zero n
    /-
      case h.e_a
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      a b : A
      n x : Nat
      hx : Membership.mem (Finset.range n.succ) x
      h : Eq (↑n.factorial) 0
      ⊢ Eq n.factorial 0
    -/
    rw [cast_eq_zero.1 h]
    /-
      🎉 no goals
    -/


/-- Shows that $e^{x} * e^{-x} = 1$ -/
theorem exp_mul_exp_neg_eq_one [Algebra ℚ A] : exp A * evalNegHom (exp A) = 1 := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    ⊢ Eq (HMul.hMul (PowerSeries.exp A) (PowerSeries.evalNegHom (PowerSeries.exp A …
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  convert exp_mul_exp_eq_exp_add (1 : A) (-1) <;> simp
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Shows that $(e^{X})^k = e^{kX}$. -/
theorem exp_pow_eq_rescale_exp [Algebra ℚ A] (k : ℕ) : exp A ^ k = rescale (k : A) (exp A) := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    k : Nat
    ⊢ Eq (HPow.hPow (PowerSeries.exp A) k) ((PowerSeries.rescale ↑k) (PowerSeries. …
  -/
  induction' k with k h
  · simp only [rescale_zero, constantCoeff_exp, Function.comp_apply, map_one, cast_zero, zero_eq,
      pow_zero (exp A), coe_comp]
  · simpa only [succ_eq_add_one, cast_add, ← exp_mul_exp_eq_exp_add (k : A), ← h, cast_one,
    id_apply, rescale_one] using pow_succ (exp A) k


/-- Shows that
$\sum_{k = 0}^{n - 1} (e^{X})^k = \sum_{p = 0}^{\infty} \sum_{k = 0}^{n - 1} \frac{k^p}{p!}X^p$. -/
theorem exp_pow_sum [Algebra ℚ A] (n : ℕ) :
    ((Finset.range n).sum fun k => exp A ^ k) =
      PowerSeries.mk fun p => (Finset.range n).sum
        fun k => (k ^ p : A) * algebraMap ℚ A p.factorial⁻¹ := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun k => HPow.hPow (PowerSeries.exp A) k) (PowerSer …
  -/
  simp only [exp_pow_eq_rescale_exp, rescale]
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun x => { toFun := fun f => PowerSeries.mk fun n = …
  -/
  ext
  simp only [one_div, coeff_mk, cast_pow, coe_mk, MonoidHom.coe_mk, OneHom.coe_mk,
    coeff_exp, factorial, map_sum]


