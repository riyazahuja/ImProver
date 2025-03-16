/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem oangle_add_right_eq_arccos_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle x (x + y) = Real.arccos (‖x‖ / ‖x + y‖) := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_add_eq_arccos_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem oangle_add_left_eq_arccos_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x + y) y = Real.arccos (‖y‖ / ‖x + y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y) ↑(Real.arccos (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)) ↑(Real.arccos (HDiv.hDiv (Norm.nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)) ↑(Real.arccos (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_add_right_eq_arccos_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem oangle_add_right_eq_arcsin_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle x (x + y) = Real.arcsin (‖y‖ / ‖x + y‖) := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_add_eq_arcsin_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem oangle_add_left_eq_arcsin_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x + y) y = Real.arcsin (‖x‖ / ‖x + y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y) ↑(Real.arcsin (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)) ↑(Real.arcsin (HDiv.hDiv (Norm.nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)) ↑(Real.arcsin (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_add_right_eq_arcsin_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem oangle_add_right_eq_arctan_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle x (x + y) = Real.arctan (‖y‖ / ‖x‖) := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_add_eq_arctan_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h) (o.left_ne_zero_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem oangle_add_left_eq_arctan_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x + y) y = Real.arctan (‖x‖ / ‖y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y) ↑(Real.arctan (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)) ↑(Real.arctan (HDiv.hDiv (Norm.nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)) ↑(Real.arctan (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_add_right_eq_arctan_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_oangle_add_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.cos (o.oangle x (x + y)) = ‖x‖ / ‖x + y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.cos_angle_add_of_inner_eq_zero (o.inner_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_oangle_add_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.cos (o.oangle (x + y) y) = ‖y‖ / ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y).cos (HDiv.hDiv (Norm.norm y) (Norm.norm (HAd …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)).cos (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)).cos (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  exact (-o).cos_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_oangle_add_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.sin (o.oangle x (x + y)) = ‖y‖ / ‖x + y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.sin_angle_add_of_inner_eq_zero (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_oangle_add_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.sin (o.oangle (x + y) y) = ‖x‖ / ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y).sin (HDiv.hDiv (Norm.norm x) (Norm.norm (HAd …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)).sin (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)).sin (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  exact (-o).sin_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_oangle_add_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.tan (o.oangle x (x + y)) = ‖y‖ / ‖x‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.tan_angle_add_of_inner_eq_zero (o.inner_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_oangle_add_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.tan (o.oangle (x + y) y) = ‖x‖ / ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y).tan (HDiv.hDiv (Norm.norm x) (Norm.norm y))
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd x y)).tan (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle y (HAdd.hAdd y x)).tan (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  exact (-o).tan_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.cos (o.oangle x (x + y)) * ‖x + y‖ = ‖x‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.cos_angle_add_mul_norm_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_oangle_add_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.cos (o.oangle (x + y) y) * ‖x + y‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HAdd.hAdd x y) y).cos (Norm.norm (HAdd.hAdd x y)))  …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd x y)).cos (Norm.norm (HAdd.hA …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd y x)).cos (Norm.norm (HAdd.hA …
  -/
  exact (-o).cos_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.sin (o.oangle x (x + y)) * ‖x + y‖ = ‖y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.sin_angle_add_mul_norm_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_oangle_add_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.sin (o.oangle (x + y) y) * ‖x + y‖ = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HAdd.hAdd x y) y).sin (Norm.norm (HAdd.hAdd x y)))  …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd x y)).sin (Norm.norm (HAdd.hA …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd y x)).sin (Norm.norm (HAdd.hA …
  -/
  exact (-o).sin_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.tan (o.oangle x (x + y)) * ‖x‖ = ‖y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.tan_angle_add_mul_norm_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_oangle_add_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.tan (o.oangle (x + y) y) * ‖y‖ = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HAdd.hAdd x y) y).tan (Norm.norm y)) (Norm.norm x)
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd x y)).tan (Norm.norm y)) (Nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle y (HAdd.hAdd y x)).tan (Norm.norm y)) (Nor …
  -/
  exact (-o).tan_oangle_add_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem norm_div_cos_oangle_add_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.cos (o.oangle x (x + y)) = ‖x + y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.norm_div_cos_angle_add_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem norm_div_cos_oangle_add_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.cos (o.oangle (x + y) y) = ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (o.oangle (HAdd.hAdd x y) y).cos) (Norm.norm (HA …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) ((Neg.neg o).oangle y (HAdd.hAdd x y)).cos) (Nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) ((Neg.neg o).oangle y (HAdd.hAdd y x)).cos) (Nor …
  -/
  exact (-o).norm_div_cos_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem norm_div_sin_oangle_add_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.sin (o.oangle x (x + y)) = ‖x + y‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.norm_div_sin_angle_add_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inr (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem norm_div_sin_oangle_add_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.sin (o.oangle (x + y) y) = ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (o.oangle (HAdd.hAdd x y) y).sin) (Norm.norm (HA …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) ((Neg.neg o).oangle y (HAdd.hAdd x y)).sin) (Nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) ((Neg.neg o).oangle y (HAdd.hAdd y x)).sin) (Nor …
  -/
  exact (-o).norm_div_sin_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem norm_div_tan_oangle_add_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.tan (o.oangle x (x + y)) = ‖x‖ := by
  have hs : (o.oangle x (x + y)).sign = 1 := by
    rw [oangle_sign_add_right, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.norm_div_tan_angle_add_of_inner_eq_zero
      (o.inner_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inr (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem norm_div_tan_oangle_add_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.tan (o.oangle (x + y) y) = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (o.oangle (HAdd.hAdd x y) y).tan) (Norm.norm y)
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) ((Neg.neg o).oangle y (HAdd.hAdd x y)).tan) (Nor …
  -/
  rw [add_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) ((Neg.neg o).oangle y (HAdd.hAdd y x)).tan) (Nor …
  -/
  exact (-o).norm_div_tan_oangle_add_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arccos`, version subtracting vectors. -/
theorem oangle_sub_right_eq_arccos_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle y (y - x) = Real.arccos (‖y‖ / ‖y - x‖) := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_sub_eq_arccos_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arccos`, version subtracting vectors. -/
theorem oangle_sub_left_eq_arccos_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x - y) x = Real.arccos (‖x‖ / ‖x - y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x) ↑(Real.arccos (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)) ↑(Real.arccos (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_sub_right_eq_arccos_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arcsin`, version subtracting vectors. -/
theorem oangle_sub_right_eq_arcsin_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle y (y - x) = Real.arcsin (‖x‖ / ‖y - x‖) := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_sub_eq_arcsin_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- An angle in a right-angled triangle expressed using `arcsin`, version subtracting vectors. -/
theorem oangle_sub_left_eq_arcsin_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x - y) x = Real.arcsin (‖y‖ / ‖x - y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x) ↑(Real.arcsin (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)) ↑(Real.arcsin (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_sub_right_eq_arcsin_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`, version subtracting vectors. -/
theorem oangle_sub_right_eq_arctan_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle y (y - x) = Real.arctan (‖x‖ / ‖y‖) := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs,
    InnerProductGeometry.angle_sub_eq_arctan_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h) (o.right_ne_zero_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arctan`, version subtracting vectors. -/
theorem oangle_sub_left_eq_arctan_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    o.oangle (x - y) x = Real.arctan (‖y‖ / ‖x‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x) ↑(Real.arctan (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)) ↑(Real.arctan (HDiv.hDiv (Norm.nor …
  -/
  exact (-o).oangle_sub_right_eq_arctan_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem cos_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.cos (o.oangle y (y - x)) = ‖y‖ / ‖y - x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.cos_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem cos_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.cos (o.oangle (x - y) x) = ‖x‖ / ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x).cos (HDiv.hDiv (Norm.norm x) (Norm.norm (HSu …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)).cos (HDiv.hDiv (Norm.norm x) (Norm …
  -/
  exact (-o).cos_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem sin_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.sin (o.oangle y (y - x)) = ‖x‖ / ‖y - x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.sin_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- The sine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem sin_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.sin (o.oangle (x - y) x) = ‖y‖ / ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x).sin (HDiv.hDiv (Norm.norm y) (Norm.norm (HSu …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)).sin (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  exact (-o).sin_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem tan_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.tan (o.oangle y (y - x)) = ‖x‖ / ‖y‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.tan_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The tangent of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem tan_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = ↑(π / 2)) :
    Real.Angle.tan (o.oangle (x - y) x) = ‖y‖ / ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (o.oangle (HSub.hSub x y) x).tan (HDiv.hDiv (Norm.norm y) (Norm.norm x))
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq ((Neg.neg o).oangle x (HSub.hSub x y)).tan (HDiv.hDiv (Norm.norm y) (Norm …
  -/
  exact (-o).tan_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side, version subtracting vectors. -/
theorem cos_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.cos (o.oangle y (y - x)) * ‖y - x‖ = ‖y‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.cos_angle_sub_mul_norm_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side, version subtracting vectors. -/
theorem cos_oangle_sub_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.cos (o.oangle (x - y) x) * ‖x - y‖ = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HSub.hSub x y) x).cos (Norm.norm (HSub.hSub x y)))  …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle x (HSub.hSub x y)).cos (Norm.norm (HSub.hS …
  -/
  exact (-o).cos_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side, version subtracting vectors. -/
theorem sin_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.sin (o.oangle y (y - x)) * ‖y - x‖ = ‖x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.sin_angle_sub_mul_norm_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)]


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side, version subtracting vectors. -/
theorem sin_oangle_sub_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.sin (o.oangle (x - y) x) * ‖x - y‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HSub.hSub x y) x).sin (Norm.norm (HSub.hSub x y)))  …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle x (HSub.hSub x y)).sin (Norm.norm (HSub.hS …
  -/
  exact (-o).sin_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side, version subtracting vectors. -/
theorem tan_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.tan (o.oangle y (y - x)) * ‖y‖ = ‖x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.tan_angle_sub_mul_norm_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side, version subtracting vectors. -/
theorem tan_oangle_sub_left_mul_norm_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : Real.Angle.tan (o.oangle (x - y) x) * ‖x‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (o.oangle (HSub.hSub x y) x).tan (Norm.norm x)) (Norm.norm y)
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul ((Neg.neg o).oangle x (HSub.hSub x y)).tan (Norm.norm x)) (Nor …
  -/
  exact (-o).tan_oangle_sub_right_mul_norm_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_cos_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.cos (o.oangle y (y - x)) = ‖y - x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    InnerProductGeometry.norm_div_cos_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inl (o.right_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_cos_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.cos (o.oangle (x - y) x) = ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (o.oangle (HSub.hSub x y) x).cos) (Norm.norm (HS …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) ((Neg.neg o).oangle x (HSub.hSub x y)).cos) (Nor …
  -/
  exact (-o).norm_div_cos_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_sin_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.sin (o.oangle y (y - x)) = ‖y - x‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    InnerProductGeometry.norm_div_sin_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inr (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_sin_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.sin (o.oangle (x - y) x) = ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (o.oangle (HSub.hSub x y) x).sin) (Norm.norm (HS …
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) ((Neg.neg o).oangle x (HSub.hSub x y)).sin) (Nor …
  -/
  exact (-o).norm_div_sin_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side, version subtracting vectors. -/
theorem norm_div_tan_oangle_sub_right_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖x‖ / Real.Angle.tan (o.oangle y (y - x)) = ‖y‖ := by
  have hs : (o.oangle y (y - x)).sign = 1 := by
    rw [oangle_sign_sub_right_swap, h, Real.Angle.sign_coe_pi_div_two]
  rw [o.oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    InnerProductGeometry.norm_div_tan_angle_sub_of_inner_eq_zero
      (o.inner_rev_eq_zero_of_oangle_eq_pi_div_two h)
      (Or.inr (o.left_ne_zero_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side, version subtracting vectors. -/
theorem norm_div_tan_oangle_sub_left_of_oangle_eq_pi_div_two {x y : V}
    (h : o.oangle x y = ↑(π / 2)) : ‖y‖ / Real.Angle.tan (o.oangle (x - y) x) = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (o.oangle (HSub.hSub x y) x).tan) (Norm.norm x)
  -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj] at h ⊢
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq ((Neg.neg o).oangle y x) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) ((Neg.neg o).oangle x (HSub.hSub x y)).tan) (Nor …
  -/
  exact (-o).norm_div_tan_oangle_sub_right_of_oangle_eq_pi_div_two h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`, where one side is a multiple
of a rotation of another by `π / 2`. -/
theorem oangle_add_right_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    o.oangle x (x + r • o.rotation (π / 2 : ℝ) x) = Real.arctan r := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle x (HAdd.hAdd x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi  …
  -/
  rcases lt_trichotomy r 0 with (hr | rfl | hr)
  · have ha : o.oangle x (r • o.rotation (π / 2 : ℝ) x) = -(π / 2 : ℝ) := by
      rw [o.oangle_smul_right_of_neg _ _ hr, o.oangle_neg_right h, o.oangle_rotation_self_right h, ←
        sub_eq_zero, add_comm, sub_neg_eq_add, ← Real.Angle.coe_add, ← Real.Angle.coe_add,
        add_assoc, add_halves, ← two_mul, Real.Angle.coe_two_pi]
      simpa using h
    -- Porting note: if the type is not given in `neg_neg` then Lean "forgets" about the instance
    -- `Neg (Orientation ℝ V (Fin 2))`
    /-
      case inl
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      hd2 : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      h : Ne x 0
      r : Real
      hr : LT.lt r 0
      ha : Eq (o.oangle x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))) ( …
      ⊢ Eq (o.oangle x (HAdd.hAdd x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi  …
    -/
    rw [← neg_inj, ← oangle_neg_orientation_eq_neg, @neg_neg Real.Angle] at ha
    rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj, oangle_rev,
      (-o).oangle_add_right_eq_arctan_of_oangle_eq_pi_div_two ha, norm_smul,
      LinearIsometryEquiv.norm_map, mul_div_assoc, div_self (norm_ne_zero_iff.2 h), mul_one,
      Real.norm_eq_abs, abs_of_neg hr, Real.arctan_neg, Real.Angle.coe_neg, neg_neg]
    /-
      case inr.inl
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      hd2 : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      h : Ne x 0
      ⊢ Eq (o.oangle x (HAdd.hAdd x (HSMul.hSMul 0 ((o.rotation ↑(HDiv.hDiv Real.pi  …
    -/
  · rw [zero_smul, add_zero, oangle_self, Real.arctan_zero, Real.Angle.coe_zero]
    /-
      🎉 no goals
    -/
  · have ha : o.oangle x (r • o.rotation (π / 2 : ℝ) x) = (π / 2 : ℝ) := by
      rw [o.oangle_smul_right_of_pos _ _ hr, o.oangle_rotation_self_right h]
    rw [o.oangle_add_right_eq_arctan_of_oangle_eq_pi_div_two ha, norm_smul,
      LinearIsometryEquiv.norm_map, mul_div_assoc, div_self (norm_ne_zero_iff.2 h), mul_one,
      Real.norm_eq_abs, abs_of_pos hr]


/-- An angle in a right-angled triangle expressed using `arctan`, where one side is a multiple
of a rotation of another by `π / 2`. -/
theorem oangle_add_left_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    o.oangle (x + r • o.rotation (π / 2 : ℝ) x) (r • o.rotation (π / 2 : ℝ) x)
      = Real.arctan r⁻¹ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle (HAdd.hAdd x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2) …
  -/
  by_cases hr : r = 0; · simp [hr]
                         /-
                           🎉 no goals
                         -/
  rw [← neg_inj, oangle_rev, ← oangle_neg_orientation_eq_neg, neg_inj, ←
    neg_neg ((π / 2 : ℝ) : Real.Angle), ← rotation_neg_orientation_eq_neg, add_comm]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    ⊢ Eq ((Neg.neg o).oangle (HSMul.hSMul r (((Neg.neg o).rotation (Neg.neg ↑(HDiv …
  -/
  have hx : x = r⁻¹ • (-o).rotation (π / 2 : ℝ) (r • (-o).rotation (-(π / 2 : ℝ)) x) := by simp [hr]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) (((Neg.neg o).rotation ↑(HDiv.hDiv Real.pi  …
    ⊢ Eq ((Neg.neg o).oangle (HSMul.hSMul r (((Neg.neg o).rotation (Neg.neg ↑(HDiv …
  -/
  nth_rw 3 [hx]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) (((Neg.neg o).rotation ↑(HDiv.hDiv Real.pi  …
    ⊢ Eq ((Neg.neg o).oangle (HSMul.hSMul r (((Neg.neg o).rotation (Neg.neg ↑(HDiv …
  -/
  refine (-o).oangle_add_right_smul_rotation_pi_div_two ?_ _
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) (((Neg.neg o).rotation ↑(HDiv.hDiv Real.pi  …
    ⊢ Ne (HSMul.hSMul r (((Neg.neg o).rotation (Neg.neg ↑(HDiv.hDiv Real.pi 2))) x …
  -/
  simp [hr, h]
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle, where one side is a multiple of a
rotation of another by `π / 2`. -/
theorem tan_oangle_add_right_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    Real.Angle.tan (o.oangle x (x + r • o.rotation (π / 2 : ℝ) x)) = r := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle x (HAdd.hAdd x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi  …
  -/
  rw [o.oangle_add_right_smul_rotation_pi_div_two h, Real.Angle.tan_coe, Real.tan_arctan]
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle, where one side is a multiple of a
rotation of another by `π / 2`. -/
theorem tan_oangle_add_left_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    Real.Angle.tan (o.oangle (x + r • o.rotation (π / 2 : ℝ) x) (r • o.rotation (π / 2 : ℝ) x)) =
      r⁻¹ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle (HAdd.hAdd x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2) …
  -/
  rw [o.oangle_add_left_smul_rotation_pi_div_two h, Real.Angle.tan_coe, Real.tan_arctan]
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`, where one side is a multiple
of a rotation of another by `π / 2`, version subtracting vectors. -/
theorem oangle_sub_right_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    o.oangle (r • o.rotation (π / 2 : ℝ) x) (r • o.rotation (π / 2 : ℝ) x - x)
      = Real.arctan r⁻¹ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) (HSub.h …
  -/
  by_cases hr : r = 0; · simp [hr]
                         /-
                           🎉 no goals
                         -/
  have hx : -x = r⁻¹ • o.rotation (π / 2 : ℝ) (r • o.rotation (π / 2 : ℝ) x) := by
    simp [hr, ← Real.Angle.coe_add]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq (Neg.neg x) (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi  …
    ⊢ Eq (o.oangle (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) (HSub.h …
  -/
  rw [sub_eq_add_neg, hx, o.oangle_add_right_smul_rotation_pi_div_two]
  /-
    case neg.h
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq (Neg.neg x) (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi  …
    ⊢ Ne (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) 0
  -/
  simpa [hr] using h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`, where one side is a multiple
of a rotation of another by `π / 2`, version subtracting vectors. -/
theorem oangle_sub_left_smul_rotation_pi_div_two {x : V} (h : x ≠ 0) (r : ℝ) :
    o.oangle (x - r • o.rotation (π / 2 : ℝ) x) x = Real.arctan r := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    ⊢ Eq (o.oangle (HSub.hSub x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2) …
  -/
  by_cases hr : r = 0; · simp [hr]
                         /-
                           🎉 no goals
                         -/
  have hx : x = r⁻¹ • o.rotation (π / 2 : ℝ) (-(r • o.rotation (π / 2 : ℝ) x)) := by
    simp [hr, ← Real.Angle.coe_add]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) (Neg.n …
    ⊢ Eq (o.oangle (HSub.hSub x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2) …
  -/
  rw [sub_eq_add_neg, add_comm]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) (Neg.n …
    ⊢ Eq (o.oangle (HAdd.hAdd (Neg.neg (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Rea …
  -/
  nth_rw 3 [hx]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) (Neg.n …
    ⊢ Eq (o.oangle (HAdd.hAdd (Neg.neg (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Rea …
  -/
  nth_rw 2 [hx]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) (Neg.n …
    ⊢ Eq (o.oangle (HAdd.hAdd (Neg.neg (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Rea …
  -/
  rw [o.oangle_add_left_smul_rotation_pi_div_two, inv_inv]
  /-
    case neg.h
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    h : Ne x 0
    r : Real
    hr : Not (Eq r 0)
    hx : Eq x (HSMul.hSMul (Inv.inv r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) (Neg.n …
    ⊢ Ne (Neg.neg (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))) 0
  -/
  simpa [hr] using h
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem oangle_right_eq_arccos_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₂ p₃ p₁ = Real.arccos (dist p₃ p₂ / dist p₁ p₃) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁) ↑(Real.arccos (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs,
    angle_eq_arccos_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem oangle_left_eq_arccos_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₃ p₁ p₂ = Real.arccos (dist p₁ p₂ / dist p₁ p₃) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂) ↑(Real.arccos (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm,
    angle_eq_arccos_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h),
    dist_comm p₁ p₃]


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem oangle_right_eq_arcsin_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₂ p₃ p₁ = Real.arcsin (dist p₁ p₂ / dist p₁ p₃) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁) ↑(Real.arcsin (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs,
    angle_eq_arcsin_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (left_ne_of_oangle_eq_pi_div_two h))]


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem oangle_left_eq_arcsin_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₃ p₁ p₂ = Real.arcsin (dist p₃ p₂ / dist p₁ p₃) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂) ↑(Real.arcsin (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm,
    angle_eq_arcsin_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (left_ne_of_oangle_eq_pi_div_two h)),
    dist_comm p₁ p₃]


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem oangle_right_eq_arctan_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₂ p₃ p₁ = Real.arctan (dist p₁ p₂ / dist p₃ p₂) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁) ↑(Real.arctan (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs,
    angle_eq_arctan_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (right_ne_of_oangle_eq_pi_div_two h)]


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem oangle_left_eq_arctan_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    ∡ p₃ p₁ p₂ = Real.arctan (dist p₃ p₂ / dist p₁ p₂) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂) ↑(Real.arctan (HDiv.hDiv (Dist.dist p …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm,
    angle_eq_arctan_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (left_ne_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.cos (∡ p₂ p₃ p₁) = dist p₃ p₂ / dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁).cos (HDiv.hDiv (Dist.dist p₃ p₂) (Dis …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    cos_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.cos (∡ p₃ p₁ p₂) = dist p₁ p₂ / dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂).cos (HDiv.hDiv (Dist.dist p₁ p₂) (Dis …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.cos_coe,
    cos_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h),
    dist_comm p₁ p₃]


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.sin (∡ p₂ p₃ p₁) = dist p₁ p₂ / dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁).sin (HDiv.hDiv (Dist.dist p₁ p₂) (Dis …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    sin_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (left_ne_of_oangle_eq_pi_div_two h))]


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.sin (∡ p₃ p₁ p₂) = dist p₃ p₂ / dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂).sin (HDiv.hDiv (Dist.dist p₃ p₂) (Dis …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.sin_coe,
    sin_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (left_ne_of_oangle_eq_pi_div_two h)),
    dist_comm p₁ p₃]


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.tan (∡ p₂ p₃ p₁) = dist p₁ p₂ / dist p₃ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₂ p₃ p₁).tan (HDiv.hDiv (Dist.dist p₁ p₂) (Dis …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    tan_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) :
    Real.Angle.tan (∡ p₃ p₁ p₂) = dist p₃ p₂ / dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (EuclideanGeometry.oangle p₃ p₁ p₂).tan (HDiv.hDiv (Dist.dist p₃ p₂) (Dis …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.tan_coe,
    tan_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_oangle_right_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.cos (∡ p₂ p₃ p₁) * dist p₁ p₃ = dist p₃ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₂ p₃ p₁).cos (Dist.dist p₁ p₃)) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    cos_angle_mul_dist_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_oangle_left_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.cos (∡ p₃ p₁ p₂) * dist p₁ p₃ = dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₃ p₁ p₂).cos (Dist.dist p₁ p₃)) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.cos_coe, dist_comm p₁ p₃,
    cos_angle_mul_dist_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_oangle_right_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.sin (∡ p₂ p₃ p₁) * dist p₁ p₃ = dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₂ p₃ p₁).sin (Dist.dist p₁ p₃)) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    sin_angle_mul_dist_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_oangle_left_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.sin (∡ p₃ p₁ p₂) * dist p₁ p₃ = dist p₃ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₃ p₁ p₂).sin (Dist.dist p₁ p₃)) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.sin_coe, dist_comm p₁ p₃,
    sin_angle_mul_dist_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)]


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_oangle_right_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.tan (∡ p₂ p₃ p₁) * dist p₃ p₂ = dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₂ p₃ p₁).tan (Dist.dist p₃ p₂)) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    tan_angle_mul_dist_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (right_ne_of_oangle_eq_pi_div_two h))]


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_oangle_left_mul_dist_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : Real.Angle.tan (∡ p₃ p₁ p₂) * dist p₁ p₂ = dist p₃ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HMul.hMul (EuclideanGeometry.oangle p₃ p₁ p₂).tan (Dist.dist p₁ p₂)) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.tan_coe,
    tan_angle_mul_dist_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (left_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem dist_div_cos_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₃ p₂ / Real.Angle.cos (∡ p₂ p₃ p₁) = dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₃ p₂) (EuclideanGeometry.oangle p₂ p₃ p₁).cos) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.cos_coe,
    dist_div_cos_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (right_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem dist_div_cos_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₁ p₂ / Real.Angle.cos (∡ p₃ p₁ p₂) = dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₁ p₂) (EuclideanGeometry.oangle p₃ p₁ p₂).cos) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.cos_coe, dist_comm p₁ p₃,
    dist_div_cos_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inr (left_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem dist_div_sin_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₁ p₂ / Real.Angle.sin (∡ p₂ p₃ p₁) = dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₁ p₂) (EuclideanGeometry.oangle p₂ p₃ p₁).sin) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.sin_coe,
    dist_div_sin_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (left_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem dist_div_sin_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₃ p₂ / Real.Angle.sin (∡ p₃ p₁ p₂) = dist p₁ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₃ p₂) (EuclideanGeometry.oangle p₃ p₁ p₂).sin) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.sin_coe, dist_comm p₁ p₃,
    dist_div_sin_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (right_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem dist_div_tan_oangle_right_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₁ p₂ / Real.Angle.tan (∡ p₂ p₃ p₁) = dist p₃ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₁ p₂) (EuclideanGeometry.oangle p₂ p₃ p₁).tan) (Di …
  -/
  have hs : (∡ p₂ p₃ p₁).sign = 1 := by rw [oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, Real.Angle.tan_coe,
    dist_div_tan_angle_of_angle_eq_pi_div_two (angle_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (left_ne_of_oangle_eq_pi_div_two h))]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem dist_div_tan_oangle_left_of_oangle_eq_pi_div_two {p₁ p₂ p₃ : P}
    (h : ∡ p₁ p₂ p₃ = ↑(π / 2)) : dist p₃ p₂ / Real.Angle.tan (∡ p₃ p₁ p₂) = dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    hd2 : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Module.Oriented Real V (Fin 2)
    p₁ p₂ p₃ : P
    h : Eq (EuclideanGeometry.oangle p₁ p₂ p₃) ↑(HDiv.hDiv Real.pi 2)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₃ p₂) (EuclideanGeometry.oangle p₃ p₁ p₂).tan) (Di …
  -/
  have hs : (∡ p₃ p₁ p₂).sign = 1 := by rw [← oangle_rotate_sign, h, Real.Angle.sign_coe_pi_div_two]
  rw [oangle_eq_angle_of_sign_eq_one hs, angle_comm, Real.Angle.tan_coe,
    dist_div_tan_angle_of_angle_eq_pi_div_two (angle_rev_eq_pi_div_two_of_oangle_eq_pi_div_two h)
      (Or.inl (right_ne_of_oangle_eq_pi_div_two h))]


