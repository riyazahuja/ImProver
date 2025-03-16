theorem injective_arg : Injective fun z : Circle => arg z := fun z w h =>
  Subtype.ext <| ext_abs_arg (z.abs_coe.trans w.abs_coe.symm) h


@[simp]
theorem arg_eq_arg {z w : Circle} : arg z = arg w ↔ z = w :=
  injective_arg.eq_iff


theorem arg_exp {x : ℝ} (h₁ : -π < x) (h₂ : x ≤ π) : arg (exp x) = x := by
  /-
    x : Real
    h₁ : LT.lt (Neg.neg Real.pi) x
    h₂ : LE.le x Real.pi
    ⊢ Eq (↑(Circle.exp x)).arg x
  -/
  rw [coe_exp, exp_mul_I, arg_cos_add_sin_mul_I ⟨h₁, h₂⟩]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_arg (z : Circle) : exp (arg z) = z :=
  injective_arg <| arg_exp (neg_pi_lt_arg _) (arg_le_pi _)


@[deprecated (since := "2024-07-25")] alias _root_.arg_expMapCircle := arg_exp

@[deprecated (since := "2024-07-25")] alias _root_.expMapCircle_arg := exp_arg


/-- `Complex.arg ∘ (↑)` and `expMapCircle` define a partial equivalence between `circle` and `ℝ`
with `source = Set.univ` and `target = Set.Ioc (-π) π`. -/
@[simps (config := .asFn)]
noncomputable def argPartialEquiv : PartialEquiv Circle ℝ where
  toFun := arg ∘ (↑)
  invFun := exp
  source := univ
  target := Ioc (-π) π
  map_source' _ _ := ⟨neg_pi_lt_arg _, arg_le_pi _⟩
  map_target' := mapsTo_univ _ _
  left_inv' z _ := exp_arg z
  right_inv' _ hx := arg_exp hx.1 hx.2


/-- `Complex.arg` and `expMapCircle` define an equivalence between `circle` and `(-π, π]`. -/
@[simps (config := .asFn)]
noncomputable def argEquiv : Circle ≃ Ioc (-π) π where
  toFun z := ⟨arg z, neg_pi_lt_arg _, arg_le_pi _⟩
  invFun := exp ∘ (↑)
  left_inv _ := argPartialEquiv.left_inv trivial
  right_inv x := Subtype.ext <| argPartialEquiv.right_inv x.2


lemma leftInverse_exp_arg : LeftInverse exp (arg ∘ (↑)) := exp_arg

lemma invOn_arg_exp : InvOn (arg ∘ (↑)) exp (Ioc (-π) π) univ := argPartialEquiv.symm.invOn

lemma surjOn_exp_neg_pi_pi : SurjOn exp (Ioc (-π) π) univ := argPartialEquiv.symm.surjOn


lemma exp_eq_exp {x y : ℝ} : exp x = exp y ↔ ∃ m : ℤ, x = y + m * (2 * π) := by
  /-
    x y : Real
    ⊢ Iff (Eq (Circle.exp x) (Circle.exp y)) (Exists fun m => Eq x (HAdd.hAdd y (H …
  -/
  rw [Subtype.ext_iff, coe_exp, coe_exp, exp_eq_exp_iff_exists_int]
  /-
    x y : Real
    ⊢ Iff (Exists fun n => Eq (HMul.hMul (↑x) Complex.I) (HAdd.hAdd (HMul.hMul (↑y …
  -/
  refine exists_congr fun n => ?_
  /-
    x y : Real
    n : Int
    ⊢ Iff (Eq (HMul.hMul (↑x) Complex.I) (HAdd.hAdd (HMul.hMul (↑y) Complex.I) (HM …
  -/
  rw [← mul_assoc, ← add_mul, mul_left_inj' I_ne_zero]
  /-
    x y : Real
    n : Int
    ⊢ Iff (Eq (↑x) (HAdd.hAdd (↑y) (HMul.hMul (↑n) (HMul.hMul 2 ↑Real.pi)))) (Eq x …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           z : Real
                                                                           ⊢ Eq (HAdd.hAdd z (HMul.hMul 2 Real.pi)) (HAdd.hAdd z (HMul.hMul (↑1) (HMul.hM …
                                                                         -/
lemma periodic_exp : Periodic exp (2 * π) := fun z ↦ exp_eq_exp.2 ⟨1, by rw [Int.cast_one, one_mul]⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma exp_two_pi : exp (2 * π) = 1 := periodic_exp.eq.trans exp_zero


lemma exp_int_mul_two_pi (n : ℤ) : exp (n * (2 * π)) = 1 :=
            /-
              n : Int
              ⊢ Eq ↑(Circle.exp (HMul.hMul (↑n) (HMul.hMul 2 Real.pi))) ↑1
            -/
  ext <| by simpa [mul_assoc] using Complex.exp_int_mul_two_pi_mul_I n
            /-
              🎉 no goals
            -/


lemma exp_two_pi_mul_int (n : ℤ) : exp (2 * π * n) = 1 := by
  /-
    n : Int
    ⊢ Eq (Circle.exp (HMul.hMul (HMul.hMul 2 Real.pi) ↑n)) 1
  -/
  simpa only [mul_comm] using exp_int_mul_two_pi n
  /-
    🎉 no goals
  -/


lemma exp_eq_one {r : ℝ} : exp r = 1 ↔ ∃ n : ℤ, r = n * (2 * π) := by
  simp [Circle.ext_iff, Complex.exp_eq_one_iff, ← mul_assoc, Complex.I_ne_zero,
    ← Complex.ofReal_inj]


lemma exp_inj {r s : ℝ} : exp r = exp s ↔ r ≡ s [PMOD (2 * π)] := by
  /-
    r s : Real
    ⊢ Iff (Eq (Circle.exp r) (Circle.exp s)) (AddCommGroup.ModEq (HMul.hMul 2 Real …
  -/
  simp [AddCommGroup.ModEq, ← exp_eq_one, div_eq_one, eq_comm (a := exp r)]
  /-
    🎉 no goals
  -/


lemma exp_sub_two_pi (x : ℝ) : exp (x - 2 * π) = exp x := periodic_exp.sub_eq x

lemma exp_add_two_pi (x : ℝ) : exp (x + 2 * π) = exp x := periodic_exp x


@[deprecated (since := "2024-07-25")]
alias _root_.leftInverse_expMapCircle_arg := leftInverse_exp_arg


@[deprecated (since := "2024-07-25")]
alias _root_.surjOn_expMapCircle_neg_pi_pi := surjOn_exp_neg_pi_pi


@[deprecated (since := "2024-07-25")] alias _root_.invOn_arg_expMapCircle := invOn_arg_exp

@[deprecated (since := "2024-07-25")] alias _root_.expMapCircle_eq_expMapCircle := exp_eq_exp

@[deprecated (since := "2024-07-25")] alias _root_.periodic_expMapCircle := periodic_exp

@[deprecated (since := "2024-07-25")] alias _root_.expMapCircle_two_pi := exp_two_pi

@[deprecated (since := "2024-07-25")] alias _root_.expMapCircle_sub_two_pi := exp_sub_two_pi

@[deprecated (since := "2024-07-25")] alias _root_.expMapCircle_add_two_pi := exp_add_two_pi


/-- `Circle.exp`, applied to a `Real.Angle`. -/
noncomputable def toCircle (θ : Angle) : Circle := Circle.periodic_exp.lift θ


@[simp] lemma toCircle_coe (x : ℝ) : toCircle x = .exp x := rfl


lemma coe_toCircle (θ : Angle) : (θ.toCircle : ℂ) = θ.cos + θ.sin * I := by
  /-
    θ : Real.Angle
    ⊢ Eq (↑θ.toCircle) (HAdd.hAdd (↑θ.cos) (HMul.hMul (↑θ.sin) Complex.I))
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (↑(↑x✝).toCircle) (HAdd.hAdd (↑(↑x✝).cos) (HMul.hMul (↑(↑x✝).sin) Complex …
  -/
  simp [exp_mul_I]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     ⊢ Eq (Real.Angle.toCircle 0) 1
                                                   -/
@[simp] lemma toCircle_zero : toCircle 0 = 1 := by rw [← coe_zero, toCircle_coe, Circle.exp_zero]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] lemma toCircle_neg (θ : Angle) : toCircle (-θ) = (toCircle θ)⁻¹ := by
  /-
    θ : Real.Angle
    ⊢ Eq (Neg.neg θ).toCircle (Inv.inv θ.toCircle)
  -/
  induction θ using Real.Angle.induction_on
  /-
    case h
    x✝ : Real
    ⊢ Eq (Neg.neg ↑x✝).toCircle (Inv.inv (↑x✝).toCircle)
  -/
  simp_rw [← coe_neg, toCircle_coe, Circle.exp_neg]
  /-
    🎉 no goals
  -/


@[simp] lemma toCircle_add (θ₁ θ₂ : Angle) : toCircle (θ₁ + θ₂) = toCircle θ₁ * toCircle θ₂ := by
  /-
    θ₁ θ₂ : Real.Angle
    ⊢ Eq (HAdd.hAdd θ₁ θ₂).toCircle (HMul.hMul θ₁.toCircle θ₂.toCircle)
  -/
  induction θ₁ using Real.Angle.induction_on
  /-
    case h
    θ₂ : Real.Angle
    x✝ : Real
    ⊢ Eq (HAdd.hAdd (↑x✝) θ₂).toCircle (HMul.hMul (↑x✝).toCircle θ₂.toCircle)
  -/
  induction θ₂ using Real.Angle.induction_on
  /-
    case h.h
    x✝¹ x✝ : Real
    ⊢ Eq (HAdd.hAdd ↑x✝¹ ↑x✝).toCircle (HMul.hMul (↑x✝¹).toCircle (↑x✝).toCircle)
  -/
  exact Circle.exp_add _ _
  /-
    🎉 no goals
  -/


@[simp] lemma arg_toCircle (θ : Real.Angle) : (arg θ.toCircle : Angle) = θ := by
  /-
    θ : Real.Angle
    ⊢ Eq (↑(↑θ.toCircle).arg) θ
  -/
  induction θ using Real.Angle.induction_on
  rw [toCircle_coe, Circle.coe_exp, exp_mul_I, ← ofReal_cos, ← ofReal_sin, ←
    Real.Angle.cos_coe, ← Real.Angle.sin_coe, arg_cos_add_sin_mul_I_coe_angle]


@[deprecated (since := "2024-07-25")] alias expMapCircle := toCircle

@[deprecated (since := "2024-07-25")] alias expMapCircle_coe := toCircle_coe

@[deprecated (since := "2024-07-25")] alias coe_expMapCircle := coe_toCircle

@[deprecated (since := "2024-07-25")] alias expMapCircle_zero := toCircle_zero

@[deprecated (since := "2024-07-25")] alias expMapCircle_neg := toCircle_neg

@[deprecated (since := "2024-07-25")] alias expMapCircle_add := toCircle_add

@[deprecated (since := "2024-07-25")] alias arg_expMapCircle := arg_toCircle


theorem scaled_exp_map_periodic : Function.Periodic (fun x => Circle.exp (2 * π / T * x)) T := by
  -- The case T = 0 is not interesting, but it is true, so we prove it to save hypotheses
  /-
    T : Real
    ⊢ Function.Periodic (fun x => Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Re …
  -/
  rcases eq_or_ne T 0 with (rfl | hT)
    /-
      case inl
      ⊢ Function.Periodic (fun x => Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Re …
    -/
  · intro x; simp
             /-
               🎉 no goals
             -/
    /-
      case inr
      T : Real
      hT : Ne T 0
      ⊢ Function.Periodic (fun x => Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Re …
    -/
  · intro x; simp_rw [mul_add]; rw [div_mul_cancel₀ _ hT, Circle.periodic_exp]
                                /-
                                  🎉 no goals
                                -/


/-- The canonical map `fun x => exp (2 π i x / T)` from `ℝ / ℤ • T` to the unit circle in `ℂ`.
If `T = 0` we understand this as the constant function 1. -/
noncomputable def toCircle : AddCircle T → Circle :=
  (@scaled_exp_map_periodic T).lift


theorem toCircle_apply_mk (x : ℝ) : @toCircle T x = Circle.exp (2 * π / T * x) :=
  rfl


theorem toCircle_add (x : AddCircle T) (y : AddCircle T) :
    @toCircle T (x + y) = toCircle x * toCircle y := by
  /-
    T : Real
    x y : AddCircle T
    ⊢ Eq (HAdd.hAdd x y).toCircle (HMul.hMul x.toCircle y.toCircle)
  -/
  induction x using QuotientAddGroup.induction_on
  /-
    case H
    T : Real
    y : AddCircle T
    z✝ : Real
    ⊢ Eq (HAdd.hAdd (↑z✝) y).toCircle (HMul.hMul (AddCircle.toCircle ↑z✝) y.toCirc …
  -/
  induction y using QuotientAddGroup.induction_on
  /-
    case H.H
    T z✝¹ z✝ : Real
    ⊢ Eq (HAdd.hAdd ↑z✝¹ ↑z✝).toCircle (HMul.hMul (AddCircle.toCircle ↑z✝¹) (AddCi …
  -/
  simp_rw [← coe_add, toCircle_apply_mk, mul_add, Circle.exp_add]
  /-
    🎉 no goals
  -/


@[simp] lemma toCircle_zero : toCircle (0 : AddCircle T) = 1 := by
  /-
    T : Real
    ⊢ Eq (AddCircle.toCircle 0) 1
  -/
  rw [← QuotientAddGroup.mk_zero, toCircle_apply_mk, mul_zero, Circle.exp_zero]
  /-
    🎉 no goals
  -/


theorem continuous_toCircle : Continuous (@toCircle T) :=
  continuous_coinduced_dom.mpr (Circle.exp.continuous.comp <| continuous_const.mul continuous_id')


theorem injective_toCircle (hT : T ≠ 0) : Function.Injective (@toCircle T) := by
  /-
    T : Real
    hT : Ne T 0
    ⊢ Function.Injective AddCircle.toCircle
  -/
  intro a b h
  /-
    T : Real
    hT : Ne T 0
    a b : AddCircle T
    h : Eq a.toCircle b.toCircle
    ⊢ Eq a b
  -/
  induction a using QuotientAddGroup.induction_on
  /-
    case H
    T : Real
    hT : Ne T 0
    b : AddCircle T
    z✝ : Real
    h : Eq (AddCircle.toCircle ↑z✝) b.toCircle
    ⊢ Eq (↑z✝) b
  -/
  induction b using QuotientAddGroup.induction_on
  /-
    case H.H
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (AddCircle.toCircle ↑z✝¹) (AddCircle.toCircle ↑z✝)
    ⊢ Eq ↑z✝¹ ↑z✝
  -/
  simp_rw [toCircle_apply_mk] at h
  /-
    case H.H
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    ⊢ Eq ↑z✝¹ ↑z✝
  -/
  obtain ⟨m, hm⟩ := Circle.exp_eq_exp.mp h.symm
  /-
    case H.H.intro
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    m : Int
    hm : Eq (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝) (HAdd.hAdd (HMul.hM …
    ⊢ Eq ↑z✝¹ ↑z✝
  -/
  rw [QuotientAddGroup.eq]; simp_rw [AddSubgroup.mem_zmultiples_iff, zsmul_eq_mul]
  /-
    case H.H.intro
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    m : Int
    hm : Eq (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝) (HAdd.hAdd (HMul.hM …
    ⊢ Exists fun k => Eq (HMul.hMul (↑k) T) (HAdd.hAdd (Neg.neg z✝¹) z✝)
  -/
  use m
  /-
    case h
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    m : Int
    hm : Eq (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝) (HAdd.hAdd (HMul.hM …
    ⊢ Eq (HMul.hMul (↑m) T) (HAdd.hAdd (Neg.neg z✝¹) z✝)
  -/
  field_simp at hm
  /-
    case h
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    m : Int
    hm : Eq (HMul.hMul (HMul.hMul 2 Real.pi) z✝) (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    ⊢ Eq (HMul.hMul (↑m) T) (HAdd.hAdd (Neg.neg z✝¹) z✝)
  -/
  rw [← mul_right_inj' Real.two_pi_pos.ne']
  /-
    case h
    T : Real
    hT : Ne T 0
    z✝¹ z✝ : Real
    h : Eq (Circle.exp (HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) T) z✝¹)) (Circl …
    m : Int
    hm : Eq (HMul.hMul (HMul.hMul 2 Real.pi) z✝) (HAdd.hAdd (HMul.hMul (HMul.hMul  …
    ⊢ Eq (HMul.hMul (HMul.hMul 2 Real.pi) (HMul.hMul (↑m) T)) (HMul.hMul (HMul.hMu …
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The homeomorphism between `AddCircle (2 * π)` and `Circle`. -/
@[simps] noncomputable def homeomorphCircle' : AddCircle (2 * π) ≃ₜ Circle where
  toFun := Angle.toCircle
  invFun := fun x ↦ arg x
  left_inv := Angle.arg_toCircle
  right_inv := Circle.exp_arg
  continuous_toFun := continuous_coinduced_dom.mpr Circle.exp.continuous
  continuous_invFun := by
    /-
      T : Real
      ⊢ Continuous { toFun := Real.Angle.toCircle, invFun := fun x => ↑(↑x).arg, lef …
    -/
    rw [continuous_iff_continuousAt]
    /-
      T : Real
      ⊢ ∀ (x : Circle), ContinuousAt { toFun := Real.Angle.toCircle, invFun := fun x …
    -/
    intro x
    /-
      T : Real
      x : Circle
      ⊢ ContinuousAt { toFun := Real.Angle.toCircle, invFun := fun x => ↑(↑x).arg, l …
    -/
    exact (continuousAt_arg_coe_angle x.coe_ne_zero).comp continuousAt_subtype_val
    /-
      🎉 no goals
    -/


theorem homeomorphCircle'_apply_mk (x : ℝ) : homeomorphCircle' x = Circle.exp x := rfl


/-- The homeomorphism between `AddCircle` and `Circle`. -/
noncomputable def homeomorphCircle (hT : T ≠ 0) : AddCircle T ≃ₜ Circle :=
                                        /-
                                          T : Real
                                          hT : Ne T 0
                                          ⊢ Ne (HMul.hMul 2 Real.pi) 0
                                        -/
  (homeomorphAddCircle T (2 * π) hT (by positivity)).trans homeomorphCircle'
                                        /-
                                          🎉 no goals
                                        -/


theorem homeomorphCircle_apply (hT : T ≠ 0) (x : AddCircle T) :
    homeomorphCircle hT x = toCircle x := by
  /-
    T : Real
    hT : Ne T 0
    x : AddCircle T
    ⊢ Eq ((AddCircle.homeomorphCircle hT) x) x.toCircle
  -/
  induction' x using QuotientAddGroup.induction_on with x
  rw [homeomorphCircle, Homeomorph.trans_apply,
    homeomorphAddCircle_apply_mk, homeomorphCircle'_apply_mk, toCircle_apply_mk]
  /-
    case H
    T : Real
    hT : Ne T 0
    x : Real
    ⊢ Eq (Circle.exp (HMul.hMul x (HMul.hMul (Inv.inv T) (HMul.hMul 2 Real.pi))))  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma isLocalHomeomorph_circleExp : IsLocalHomeomorph Circle.exp := by
  /-
    ⊢ IsLocalHomeomorph ⇑Circle.exp
  -/
  have : Fact (0 < 2 * π) := ⟨by positivity⟩
  /-
    this : Fact (LT.lt 0 (HMul.hMul 2 Real.pi))
    ⊢ IsLocalHomeomorph ⇑Circle.exp
  -/
  exact homeomorphCircle'.isLocalHomeomorph.comp (isLocalHomeomorph_coe (2 * π))
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-25")]
alias isLocalHomeomorph_expMapCircle := isLocalHomeomorph_circleExp

