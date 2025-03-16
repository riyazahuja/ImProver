local notation "ω" => o.areaForm


/-- The oriented angle from `x` to `y`, modulo `2 * π`. If either vector is 0, this is 0.
See `InnerProductGeometry.angle` for the corresponding unoriented angle definition. -/
def oangle (x y : V) : Real.Angle :=
  Complex.arg (o.kahler x y)


/-- Oriented angles are continuous when the vectors involved are nonzero. -/
theorem continuousAt_oangle {x : V × V} (hx1 : x.1 ≠ 0) (hx2 : x.2 ≠ 0) :
    ContinuousAt (fun y : V × V => o.oangle y.1 y.2) x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : Prod V V
    hx1 : Ne x.1 0
    hx2 : Ne x.2 0
    ⊢ ContinuousAt (fun y => o.oangle y.1 y.2) x
  -/
  refine (Complex.continuousAt_arg_coe_angle ?_).comp ?_
    /-
      case refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : Prod V V
      hx1 : Ne x.1 0
      hx2 : Ne x.2 0
      ⊢ Ne ((o.kahler x.1) x.2) 0
    -/
  · exact o.kahler_ne_zero hx1 hx2
    /-
      🎉 no goals
    -/
  exact ((continuous_ofReal.comp continuous_inner).add
    ((continuous_ofReal.comp o.areaForm'.continuous₂).mul continuous_const)).continuousAt


/-- If the first vector passed to `oangle` is 0, the result is 0. -/
@[simp]
                                                          /-
                                                            V : Type u_1
                                                            inst✝² : NormedAddCommGroup V
                                                            inst✝¹ : InnerProductSpace Real V
                                                            inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                            o : Orientation Real V (Fin 2)
                                                            x : V
                                                            ⊢ Eq (o.oangle 0 x) 0
                                                          -/
theorem oangle_zero_left (x : V) : o.oangle 0 x = 0 := by simp [oangle]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- If the second vector passed to `oangle` is 0, the result is 0. -/
@[simp]
                                                           /-
                                                             V : Type u_1
                                                             inst✝² : NormedAddCommGroup V
                                                             inst✝¹ : InnerProductSpace Real V
                                                             inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                             o : Orientation Real V (Fin 2)
                                                             x : V
                                                             ⊢ Eq (o.oangle x 0) 0
                                                           -/
theorem oangle_zero_right (x : V) : o.oangle x 0 = 0 := by simp [oangle]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- If the two vectors passed to `oangle` are the same, the result is 0. -/
@[simp]
theorem oangle_self (x : V) : o.oangle x x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (o.oangle x x) 0
  -/
  rw [oangle, kahler_apply_self, ← ofReal_pow]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (↑(↑(HPow.hPow (Norm.norm x) 2)).arg) 0
  -/
  convert QuotientAddGroup.mk_zero (AddSubgroup.zmultiples (2 * π))
  /-
    case h.e'_2.h.h.e'_1
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    e_1✝ : Eq Real.Angle (HasQuotient.Quotient Real (AddSubgroup.zmultiples (HMul. …
    ⊢ Eq (↑(HPow.hPow (Norm.norm x) 2)).arg 0
  -/
  apply arg_ofReal_of_nonneg
  /-
    case h.e'_2.h.h.e'_1.hx
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    e_1✝ : Eq Real.Angle (HasQuotient.Quotient Real (AddSubgroup.zmultiples (HMul. …
    ⊢ LE.le 0 (HPow.hPow (Norm.norm x) 2)
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is nonzero, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_ne_zero {x y : V} (h : o.oangle x y ≠ 0) : x ≠ 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Ne (o.oangle x y) 0
    ⊢ Ne x 0
  -/
  rintro rfl; simp at h
              /-
                🎉 no goals
              -/


/-- If the angle between two vectors is nonzero, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_ne_zero {x y : V} (h : o.oangle x y ≠ 0) : y ≠ 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Ne (o.oangle x y) 0
    ⊢ Ne y 0
  -/
  rintro rfl; simp at h
              /-
                🎉 no goals
              -/


/-- If the angle between two vectors is nonzero, the vectors are not equal. -/
theorem ne_of_oangle_ne_zero {x y : V} (h : o.oangle x y ≠ 0) : x ≠ y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Ne (o.oangle x y) 0
    ⊢ Ne x y
  -/
  rintro rfl; simp at h
              /-
                🎉 no goals
              -/


/-- If the angle between two vectors is `π`, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_eq_pi {x y : V} (h : o.oangle x y = π) : x ≠ 0 :=
  o.left_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `π`, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_eq_pi {x y : V} (h : o.oangle x y = π) : y ≠ 0 :=
  o.right_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `π`, the vectors are not equal. -/
theorem ne_of_oangle_eq_pi {x y : V} (h : o.oangle x y = π) : x ≠ y :=
  o.ne_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `π / 2`, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = (π / 2 : ℝ)) : x ≠ 0 :=
  o.left_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `π / 2`, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = (π / 2 : ℝ)) : y ≠ 0 :=
  o.right_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `π / 2`, the vectors are not equal. -/
theorem ne_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = (π / 2 : ℝ)) : x ≠ y :=
  o.ne_of_oangle_ne_zero (h.symm ▸ Real.Angle.pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `-π / 2`, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_eq_neg_pi_div_two {x y : V} (h : o.oangle x y = (-π / 2 : ℝ)) :
    x ≠ 0 :=
  o.left_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.neg_pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `-π / 2`, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_eq_neg_pi_div_two {x y : V} (h : o.oangle x y = (-π / 2 : ℝ)) :
    y ≠ 0 :=
  o.right_ne_zero_of_oangle_ne_zero (h.symm ▸ Real.Angle.neg_pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the angle between two vectors is `-π / 2`, the vectors are not equal. -/
theorem ne_of_oangle_eq_neg_pi_div_two {x y : V} (h : o.oangle x y = (-π / 2 : ℝ)) : x ≠ y :=
  o.ne_of_oangle_ne_zero (h.symm ▸ Real.Angle.neg_pi_div_two_ne_zero : o.oangle x y ≠ 0)


/-- If the sign of the angle between two vectors is nonzero, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_sign_ne_zero {x y : V} (h : (o.oangle x y).sign ≠ 0) : x ≠ 0 :=
  o.left_ne_zero_of_oangle_ne_zero (Real.Angle.sign_ne_zero_iff.1 h).1


/-- If the sign of the angle between two vectors is nonzero, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_sign_ne_zero {x y : V} (h : (o.oangle x y).sign ≠ 0) : y ≠ 0 :=
  o.right_ne_zero_of_oangle_ne_zero (Real.Angle.sign_ne_zero_iff.1 h).1


/-- If the sign of the angle between two vectors is nonzero, the vectors are not equal. -/
theorem ne_of_oangle_sign_ne_zero {x y : V} (h : (o.oangle x y).sign ≠ 0) : x ≠ y :=
  o.ne_of_oangle_ne_zero (Real.Angle.sign_ne_zero_iff.1 h).1


/-- If the sign of the angle between two vectors is positive, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_sign_eq_one {x y : V} (h : (o.oangle x y).sign = 1) : x ≠ 0 :=
                                                     /-
                                                       V : Type u_1
                                                       inst✝² : NormedAddCommGroup V
                                                       inst✝¹ : InnerProductSpace Real V
                                                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                       o : Orientation Real V (Fin 2)
                                                       x y : V
                                                       h : Eq (o.oangle x y).sign 1
                                                       ⊢ Ne 1 0
                                                     -/
  o.left_ne_zero_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If the sign of the angle between two vectors is positive, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_sign_eq_one {x y : V} (h : (o.oangle x y).sign = 1) : y ≠ 0 :=
                                                      /-
                                                        V : Type u_1
                                                        inst✝² : NormedAddCommGroup V
                                                        inst✝¹ : InnerProductSpace Real V
                                                        inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                        o : Orientation Real V (Fin 2)
                                                        x y : V
                                                        h : Eq (o.oangle x y).sign 1
                                                        ⊢ Ne 1 0
                                                      -/
  o.right_ne_zero_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If the sign of the angle between two vectors is positive, the vectors are not equal. -/
theorem ne_of_oangle_sign_eq_one {x y : V} (h : (o.oangle x y).sign = 1) : x ≠ y :=
                                           /-
                                             V : Type u_1
                                             inst✝² : NormedAddCommGroup V
                                             inst✝¹ : InnerProductSpace Real V
                                             inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                             o : Orientation Real V (Fin 2)
                                             x y : V
                                             h : Eq (o.oangle x y).sign 1
                                             ⊢ Ne 1 0
                                           -/
  o.ne_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                           /-
                                             🎉 no goals
                                           -/


/-- If the sign of the angle between two vectors is negative, the first vector is nonzero. -/
theorem left_ne_zero_of_oangle_sign_eq_neg_one {x y : V} (h : (o.oangle x y).sign = -1) : x ≠ 0 :=
                                                     /-
                                                       V : Type u_1
                                                       inst✝² : NormedAddCommGroup V
                                                       inst✝¹ : InnerProductSpace Real V
                                                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                       o : Orientation Real V (Fin 2)
                                                       x y : V
                                                       h : Eq (o.oangle x y).sign (-1)
                                                       ⊢ Ne (-1) 0
                                                     -/
  o.left_ne_zero_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If the sign of the angle between two vectors is negative, the second vector is nonzero. -/
theorem right_ne_zero_of_oangle_sign_eq_neg_one {x y : V} (h : (o.oangle x y).sign = -1) : y ≠ 0 :=
                                                      /-
                                                        V : Type u_1
                                                        inst✝² : NormedAddCommGroup V
                                                        inst✝¹ : InnerProductSpace Real V
                                                        inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                        o : Orientation Real V (Fin 2)
                                                        x y : V
                                                        h : Eq (o.oangle x y).sign (-1)
                                                        ⊢ Ne (-1) 0
                                                      -/
  o.right_ne_zero_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If the sign of the angle between two vectors is negative, the vectors are not equal. -/
theorem ne_of_oangle_sign_eq_neg_one {x y : V} (h : (o.oangle x y).sign = -1) : x ≠ y :=
                                           /-
                                             V : Type u_1
                                             inst✝² : NormedAddCommGroup V
                                             inst✝¹ : InnerProductSpace Real V
                                             inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                             o : Orientation Real V (Fin 2)
                                             x y : V
                                             h : Eq (o.oangle x y).sign (-1)
                                             ⊢ Ne (-1) 0
                                           -/
  o.ne_of_oangle_sign_ne_zero (h.symm ▸ by decide : (o.oangle x y).sign ≠ 0)
                                           /-
                                             🎉 no goals
                                           -/


/-- Swapping the two vectors passed to `oangle` negates the angle. -/
theorem oangle_rev (x y : V) : o.oangle y x = -o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle y x) (Neg.neg (o.oangle x y))
  -/
  simp only [oangle, o.kahler_swap y x, Complex.arg_conj_coe_angle]
  /-
    🎉 no goals
  -/


/-- Adding the angles between two vectors in each order results in 0. -/
@[simp]
theorem oangle_add_oangle_rev (x y : V) : o.oangle x y + o.oangle y x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (HAdd.hAdd (o.oangle x y) (o.oangle y x)) 0
  -/
  simp [o.oangle_rev y x]
  /-
    🎉 no goals
  -/


/-- Negating the first vector passed to `oangle` adds `π` to the angle. -/
theorem oangle_neg_left {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    o.oangle (-x) y = o.oangle x y + π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (o.oangle (Neg.neg x) y) (HAdd.hAdd (o.oangle x y) ↑Real.pi)
  -/
  simp only [oangle, map_neg]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (↑((Neg.neg (o.kahler x)) y).arg) (HAdd.hAdd ↑((o.kahler x) y).arg ↑Real. …
  -/
  convert Complex.arg_neg_coe_angle _
  /-
    case convert_2
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Ne ((o.kahler x) y) 0
  -/
  exact o.kahler_ne_zero hx hy
  /-
    🎉 no goals
  -/


/-- Negating the second vector passed to `oangle` adds `π` to the angle. -/
theorem oangle_neg_right {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    o.oangle x (-y) = o.oangle x y + π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (o.oangle x (Neg.neg y)) (HAdd.hAdd (o.oangle x y) ↑Real.pi)
  -/
  simp only [oangle, map_neg]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (↑(Neg.neg ((o.kahler x) y)).arg) (HAdd.hAdd ↑((o.kahler x) y).arg ↑Real. …
  -/
  convert Complex.arg_neg_coe_angle _
  /-
    case convert_2
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Ne ((o.kahler x) y) 0
  -/
  exact o.kahler_ne_zero hx hy
  /-
    🎉 no goals
  -/


/-- Negating the first vector passed to `oangle` does not change twice the angle. -/
@[simp]
theorem two_zsmul_oangle_neg_left (x y : V) :
    (2 : ℤ) • o.oangle (-x) y = (2 : ℤ) • o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) y)) (HSMul.hSMul 2 (o.oangle x y))
  -/
  by_cases hx : x = 0
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Eq x 0
      ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) y)) (HSMul.hSMul 2 (o.oangle x y))
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) y)) (HSMul.hSMul 2 (o.oangle x y))
    -/
  · by_cases hy : y = 0
      /-
        case pos
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) y)) (HSMul.hSMul 2 (o.oangle x y))
      -/
    · simp [hy]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) y)) (HSMul.hSMul 2 (o.oangle x y))
      -/
    · simp [o.oangle_neg_left hx hy]
      /-
        🎉 no goals
      -/


/-- Negating the second vector passed to `oangle` does not change twice the angle. -/
@[simp]
theorem two_zsmul_oangle_neg_right (x y : V) :
    (2 : ℤ) • o.oangle x (-y) = (2 : ℤ) • o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg y))) (HSMul.hSMul 2 (o.oangle x y))
  -/
  by_cases hx : x = 0
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Eq x 0
      ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg y))) (HSMul.hSMul 2 (o.oangle x y))
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg y))) (HSMul.hSMul 2 (o.oangle x y))
    -/
  · by_cases hy : y = 0
      /-
        case pos
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg y))) (HSMul.hSMul 2 (o.oangle x y))
      -/
    · simp [hy]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg y))) (HSMul.hSMul 2 (o.oangle x y))
      -/
    · simp [o.oangle_neg_right hx hy]
      /-
        🎉 no goals
      -/


/-- Negating both vectors passed to `oangle` does not change the angle. -/
@[simp]
                                                                           /-
                                                                             V : Type u_1
                                                                             inst✝² : NormedAddCommGroup V
                                                                             inst✝¹ : InnerProductSpace Real V
                                                                             inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                                             o : Orientation Real V (Fin 2)
                                                                             x y : V
                                                                             ⊢ Eq (o.oangle (Neg.neg x) (Neg.neg y)) (o.oangle x y)
                                                                           -/
theorem oangle_neg_neg (x y : V) : o.oangle (-x) (-y) = o.oangle x y := by simp [oangle]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Negating the first vector produces the same angle as negating the second vector. -/
theorem oangle_neg_left_eq_neg_right (x y : V) : o.oangle (-x) y = o.oangle x (-y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (Neg.neg x) y) (o.oangle x (Neg.neg y))
  -/
  rw [← neg_neg y, oangle_neg_neg, neg_neg]
  /-
    🎉 no goals
  -/


/-- The angle between the negation of a nonzero vector and that vector is `π`. -/
@[simp]
theorem oangle_neg_self_left {x : V} (hx : x ≠ 0) : o.oangle (-x) x = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    hx : Ne x 0
    ⊢ Eq (o.oangle (Neg.neg x) x) ↑Real.pi
  -/
  simp [oangle_neg_left, hx]
  /-
    🎉 no goals
  -/


/-- The angle between a nonzero vector and its negation is `π`. -/
@[simp]
theorem oangle_neg_self_right {x : V} (hx : x ≠ 0) : o.oangle x (-x) = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    hx : Ne x 0
    ⊢ Eq (o.oangle x (Neg.neg x)) ↑Real.pi
  -/
  simp [oangle_neg_right, hx]
  /-
    🎉 no goals
  -/


/-- Twice the angle between the negation of a vector and that vector is 0. -/
theorem two_zsmul_oangle_neg_self_left (x : V) : (2 : ℤ) • o.oangle (-x) x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (HSMul.hSMul 2 (o.oangle (Neg.neg x) x)) 0
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x = 0 <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


/-- Twice the angle between a vector and its negation is 0. -/
theorem two_zsmul_oangle_neg_self_right (x : V) : (2 : ℤ) • o.oangle x (-x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x (Neg.neg x))) 0
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x = 0 <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


/-- Adding the angles between two vectors in each order, with the first vector in each angle
negated, results in 0. -/
@[simp]
theorem oangle_add_oangle_rev_neg_left (x y : V) : o.oangle (-x) y + o.oangle (-y) x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (HAdd.hAdd (o.oangle (Neg.neg x) y) (o.oangle (Neg.neg y) x)) 0
  -/
  rw [oangle_neg_left_eq_neg_right, oangle_rev, neg_add_cancel]
  /-
    🎉 no goals
  -/


/-- Adding the angles between two vectors in each order, with the second vector in each angle
negated, results in 0. -/
@[simp]
theorem oangle_add_oangle_rev_neg_right (x y : V) : o.oangle x (-y) + o.oangle y (-x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (HAdd.hAdd (o.oangle x (Neg.neg y)) (o.oangle y (Neg.neg x))) 0
  -/
  rw [o.oangle_rev (-x), oangle_neg_left_eq_neg_right, add_neg_cancel]
  /-
    🎉 no goals
  -/


/-- Multiplying the first vector passed to `oangle` by a positive real does not change the
angle. -/
@[simp]
theorem oangle_smul_left_of_pos (x y : V) {r : ℝ} (hr : 0 < r) :
                                            /-
                                              V : Type u_1
                                              inst✝² : NormedAddCommGroup V
                                              inst✝¹ : InnerProductSpace Real V
                                              inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                              o : Orientation Real V (Fin 2)
                                              x y : V
                                              r : Real
                                              hr : LT.lt 0 r
                                              ⊢ Eq (o.oangle (HSMul.hSMul r x) y) (o.oangle x y)
                                            -/
    o.oangle (r • x) y = o.oangle x y := by simp [oangle, Complex.arg_real_mul _ hr]
                                            /-
                                              🎉 no goals
                                            -/


/-- Multiplying the second vector passed to `oangle` by a positive real does not change the
angle. -/
@[simp]
theorem oangle_smul_right_of_pos (x y : V) {r : ℝ} (hr : 0 < r) :
                                            /-
                                              V : Type u_1
                                              inst✝² : NormedAddCommGroup V
                                              inst✝¹ : InnerProductSpace Real V
                                              inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                              o : Orientation Real V (Fin 2)
                                              x y : V
                                              r : Real
                                              hr : LT.lt 0 r
                                              ⊢ Eq (o.oangle x (HSMul.hSMul r y)) (o.oangle x y)
                                            -/
    o.oangle x (r • y) = o.oangle x y := by simp [oangle, Complex.arg_real_mul _ hr]
                                            /-
                                              🎉 no goals
                                            -/


/-- Multiplying the first vector passed to `oangle` by a negative real produces the same angle
as negating that vector. -/
@[simp]
theorem oangle_smul_left_of_neg (x y : V) {r : ℝ} (hr : r < 0) :
    o.oangle (r • x) y = o.oangle (-x) y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (o.oangle (HSMul.hSMul r x) y) (o.oangle (Neg.neg x) y)
  -/
  rw [← neg_neg r, neg_smul, ← smul_neg, o.oangle_smul_left_of_pos _ _ (neg_pos_of_neg hr)]
  /-
    🎉 no goals
  -/


/-- Multiplying the second vector passed to `oangle` by a negative real produces the same angle
as negating that vector. -/
@[simp]
theorem oangle_smul_right_of_neg (x y : V) {r : ℝ} (hr : r < 0) :
    o.oangle x (r • y) = o.oangle x (-y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (o.oangle x (HSMul.hSMul r y)) (o.oangle x (Neg.neg y))
  -/
  rw [← neg_neg r, neg_smul, ← smul_neg, o.oangle_smul_right_of_pos _ _ (neg_pos_of_neg hr)]
  /-
    🎉 no goals
  -/


/-- The angle between a nonnegative multiple of a vector and that vector is 0. -/
@[simp]
theorem oangle_smul_left_self_of_nonneg (x : V) {r : ℝ} (hr : 0 ≤ r) : o.oangle (r • x) x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (o.oangle (HSMul.hSMul r x) x) 0
  -/
  rcases hr.lt_or_eq with (h | h)
    /-
      case inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      hr : LE.le 0 r
      h : LT.lt 0 r
      ⊢ Eq (o.oangle (HSMul.hSMul r x) x) 0
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      hr : LE.le 0 r
      h : Eq 0 r
      ⊢ Eq (o.oangle (HSMul.hSMul r x) x) 0
    -/
  · simp [h.symm]
    /-
      🎉 no goals
    -/


/-- The angle between a vector and a nonnegative multiple of that vector is 0. -/
@[simp]
theorem oangle_smul_right_self_of_nonneg (x : V) {r : ℝ} (hr : 0 ≤ r) : o.oangle x (r • x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (o.oangle x (HSMul.hSMul r x)) 0
  -/
  rcases hr.lt_or_eq with (h | h)
    /-
      case inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      hr : LE.le 0 r
      h : LT.lt 0 r
      ⊢ Eq (o.oangle x (HSMul.hSMul r x)) 0
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      hr : LE.le 0 r
      h : Eq 0 r
      ⊢ Eq (o.oangle x (HSMul.hSMul r x)) 0
    -/
  · simp [h.symm]
    /-
      🎉 no goals
    -/


/-- The angle between two nonnegative multiples of the same vector is 0. -/
@[simp]
theorem oangle_smul_smul_self_of_nonneg (x : V) {r₁ r₂ : ℝ} (hr₁ : 0 ≤ r₁) (hr₂ : 0 ≤ r₂) :
    o.oangle (r₁ • x) (r₂ • x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r₁ r₂ : Real
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le 0 r₂
    ⊢ Eq (o.oangle (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ x)) 0
  -/
  rcases hr₁.lt_or_eq with (h | h)
    /-
      case inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r₁ r₂ : Real
      hr₁ : LE.le 0 r₁
      hr₂ : LE.le 0 r₂
      h : LT.lt 0 r₁
      ⊢ Eq (o.oangle (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ x)) 0
    -/
  · simp [h, hr₂]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r₁ r₂ : Real
      hr₁ : LE.le 0 r₁
      hr₂ : LE.le 0 r₂
      h : Eq 0 r₁
      ⊢ Eq (o.oangle (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ x)) 0
    -/
  · simp [h.symm]
    /-
      🎉 no goals
    -/


/-- Multiplying the first vector passed to `oangle` by a nonzero real does not change twice the
angle. -/
@[simp]
theorem two_zsmul_oangle_smul_left_of_ne_zero (x y : V) {r : ℝ} (hr : r ≠ 0) :
    (2 : ℤ) • o.oangle (r • x) y = (2 : ℤ) • o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    hr : Ne r 0
    ⊢ Eq (HSMul.hSMul 2 (o.oangle (HSMul.hSMul r x) y)) (HSMul.hSMul 2 (o.oangle x …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rcases hr.lt_or_lt with (h | h) <;> simp [h]
                                      /-
                                        🎉 no goals
                                      -/


/-- Multiplying the second vector passed to `oangle` by a nonzero real does not change twice the
angle. -/
@[simp]
theorem two_zsmul_oangle_smul_right_of_ne_zero (x y : V) {r : ℝ} (hr : r ≠ 0) :
    (2 : ℤ) • o.oangle x (r • y) = (2 : ℤ) • o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    hr : Ne r 0
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x (HSMul.hSMul r y))) (HSMul.hSMul 2 (o.oangle x …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rcases hr.lt_or_lt with (h | h) <;> simp [h]
                                      /-
                                        🎉 no goals
                                      -/


/-- Twice the angle between a multiple of a vector and that vector is 0. -/
@[simp]
theorem two_zsmul_oangle_smul_left_self (x : V) {r : ℝ} : (2 : ℤ) • o.oangle (r • x) x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (HSMul.hSMul 2 (o.oangle (HSMul.hSMul r x) x)) 0
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases lt_or_le r 0 with (h | h) <;> simp [h]
                                       /-
                                         🎉 no goals
                                       -/


/-- Twice the angle between a vector and a multiple of that vector is 0. -/
@[simp]
theorem two_zsmul_oangle_smul_right_self (x : V) {r : ℝ} : (2 : ℤ) • o.oangle x (r • x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x (HSMul.hSMul r x))) 0
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases lt_or_le r 0 with (h | h) <;> simp [h]
                                       /-
                                         🎉 no goals
                                       -/


/-- Twice the angle between two multiples of a vector is 0. -/
@[simp]
theorem two_zsmul_oangle_smul_smul_self (x : V) {r₁ r₂ : ℝ} :
                                                   /-
                                                     V : Type u_1
                                                     inst✝² : NormedAddCommGroup V
                                                     inst✝¹ : InnerProductSpace Real V
                                                     inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                     o : Orientation Real V (Fin 2)
                                                     x : V
                                                     r₁ r₂ : Real
                                                     ⊢ Eq (HSMul.hSMul 2 (o.oangle (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ x))) 0
                                                   -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    (2 : ℤ) • o.oangle (r₁ • x) (r₂ • x) = 0 := by by_cases h : r₁ = 0 <;> simp [h]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- If the spans of two vectors are equal, twice angles with those vectors on the left are
equal. -/
theorem two_zsmul_oangle_left_of_span_eq {x y : V} (z : V) (h : (ℝ ∙ x) = ℝ ∙ y) :
    (2 : ℤ) • o.oangle x z = (2 : ℤ) • o.oangle y z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    h : Eq (Submodule.span Real (Singleton.singleton x)) (Submodule.span Real (Sin …
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x z)) (HSMul.hSMul 2 (o.oangle y z))
  -/
  rw [Submodule.span_singleton_eq_span_singleton] at h
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    h : Exists fun z => Eq (HSMul.hSMul z x) y
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x z)) (HSMul.hSMul 2 (o.oangle y z))
  -/
  rcases h with ⟨r, rfl⟩
  /-
    case intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x z : V
    r : Units Real
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x z)) (HSMul.hSMul 2 (o.oangle (HSMul.hSMul r x) …
  -/
  exact (o.two_zsmul_oangle_smul_left_of_ne_zero _ _ (Units.ne_zero _)).symm
  /-
    🎉 no goals
  -/


/-- If the spans of two vectors are equal, twice angles with those vectors on the right are
equal. -/
theorem two_zsmul_oangle_right_of_span_eq (x : V) {y z : V} (h : (ℝ ∙ y) = ℝ ∙ z) :
    (2 : ℤ) • o.oangle x y = (2 : ℤ) • o.oangle x z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    h : Eq (Submodule.span Real (Singleton.singleton y)) (Submodule.span Real (Sin …
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x y)) (HSMul.hSMul 2 (o.oangle x z))
  -/
  rw [Submodule.span_singleton_eq_span_singleton] at h
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    h : Exists fun z_1 => Eq (HSMul.hSMul z_1 y) z
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x y)) (HSMul.hSMul 2 (o.oangle x z))
  -/
  rcases h with ⟨r, rfl⟩
  /-
    case intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Units Real
    ⊢ Eq (HSMul.hSMul 2 (o.oangle x y)) (HSMul.hSMul 2 (o.oangle x (HSMul.hSMul r  …
  -/
  exact (o.two_zsmul_oangle_smul_right_of_ne_zero _ _ (Units.ne_zero _)).symm
  /-
    🎉 no goals
  -/


/-- If the spans of two pairs of vectors are equal, twice angles between those vectors are
equal. -/
theorem two_zsmul_oangle_of_span_eq_of_span_eq {w x y z : V} (hwx : (ℝ ∙ w) = ℝ ∙ x)
    (hyz : (ℝ ∙ y) = ℝ ∙ z) : (2 : ℤ) • o.oangle w y = (2 : ℤ) • o.oangle x z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    w x y z : V
    hwx : Eq (Submodule.span Real (Singleton.singleton w)) (Submodule.span Real (S …
    hyz : Eq (Submodule.span Real (Singleton.singleton y)) (Submodule.span Real (S …
    ⊢ Eq (HSMul.hSMul 2 (o.oangle w y)) (HSMul.hSMul 2 (o.oangle x z))
  -/
  rw [o.two_zsmul_oangle_left_of_span_eq y hwx, o.two_zsmul_oangle_right_of_span_eq x hyz]
  /-
    🎉 no goals
  -/


/-- The oriented angle between two vectors is zero if and only if the angle with the vectors
swapped is zero. -/
theorem oangle_eq_zero_iff_oangle_rev_eq_zero {x y : V} : o.oangle x y = 0 ↔ o.oangle y x = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (o.oangle x y) 0) (Eq (o.oangle y x) 0)
  -/
  rw [oangle_rev, neg_eq_zero]
  /-
    🎉 no goals
  -/


/-- The oriented angle between two vectors is zero if and only if they are on the same ray. -/
theorem oangle_eq_zero_iff_sameRay {x y : V} : o.oangle x y = 0 ↔ SameRay ℝ x y := by
  rw [oangle, kahler_apply_apply, Complex.arg_coe_angle_eq_iff_eq_toReal, Real.Angle.toReal_zero,
    Complex.arg_eq_zero_iff]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (And (LE.le 0 (HAdd.hAdd (↑(Inner.inner x y)) (HSMul.hSMul ((o.areaForm  …
  -/
  simpa using o.nonneg_inner_and_areaForm_eq_zero_iff_sameRay x y
  /-
    🎉 no goals
  -/


/-- The oriented angle between two vectors is `π` if and only if the angle with the vectors
swapped is `π`. -/
theorem oangle_eq_pi_iff_oangle_rev_eq_pi {x y : V} : o.oangle x y = π ↔ o.oangle y x = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (Eq (o.oangle y x) ↑Real.pi)
  -/
  rw [oangle_rev, neg_eq_iff_eq_neg, Real.Angle.neg_coe_pi]
  /-
    🎉 no goals
  -/


/-- The oriented angle between two vectors is `π` if and only they are nonzero and the first is
on the same ray as the negation of the second. -/
theorem oangle_eq_pi_iff_sameRay_neg {x y : V} :
    o.oangle x y = π ↔ x ≠ 0 ∧ y ≠ 0 ∧ SameRay ℝ x (-y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (And (Ne x 0) (And (Ne y 0) (SameRay Real x …
  -/
  rw [← o.oangle_eq_zero_iff_sameRay]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ Eq (o.oangle x y) ↑Real.pi → And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x (Neg …
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (o.oangle x y) ↑Real.pi
      ⊢ And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x (Neg.neg y)) 0))
    -/
    by_cases hx : x = 0; · simp [hx, Real.Angle.pi_ne_zero.symm] at h
                           /-
                             🎉 no goals
                           -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (o.oangle x y) ↑Real.pi
      hx : Not (Eq x 0)
      ⊢ And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x (Neg.neg y)) 0))
    -/
    by_cases hy : y = 0; · simp [hy, Real.Angle.pi_ne_zero.symm] at h
                           /-
                             🎉 no goals
                           -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (o.oangle x y) ↑Real.pi
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      ⊢ And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x (Neg.neg y)) 0))
    -/
    refine ⟨hx, hy, ?_⟩
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (o.oangle x y) ↑Real.pi
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      ⊢ Eq (o.oangle x (Neg.neg y)) 0
    -/
    rw [o.oangle_neg_right hx hy, h, Real.Angle.coe_pi_add_coe_pi]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ And (Ne x 0) (And (Ne y 0) (Eq (o.oangle x (Neg.neg y)) 0)) → Eq (o.oangle x …
    -/
  · rintro ⟨hx, hy, h⟩
    /-
      case mpr.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (o.oangle x (Neg.neg y)) 0
      ⊢ Eq (o.oangle x y) ↑Real.pi
    -/
    rwa [o.oangle_neg_right hx hy, ← Real.Angle.sub_coe_pi_eq_add_coe_pi, sub_eq_zero] at h
    /-
      🎉 no goals
    -/


/-- The oriented angle between two vectors is zero or `π` if and only if those two vectors are
not linearly independent. -/
theorem oangle_eq_zero_or_eq_pi_iff_not_linearIndependent {x y : V} :
    o.oangle x y = 0 ∨ o.oangle x y = π ↔ ¬LinearIndependent ℝ ![x, y] := by
  rw [oangle_eq_zero_iff_sameRay, oangle_eq_pi_iff_sameRay_neg,
    sameRay_or_ne_zero_and_sameRay_neg_iff_not_linearIndependent]


/-- The oriented angle between two vectors is zero or `π` if and only if the first vector is zero
or the second is a multiple of the first. -/
theorem oangle_eq_zero_or_eq_pi_iff_right_eq_smul {x y : V} :
    o.oangle x y = 0 ∨ o.oangle x y = π ↔ x = 0 ∨ ∃ r : ℝ, y = r • x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Or (Eq (o.oangle x y) 0) (Eq (o.oangle x y) ↑Real.pi)) (Or (Eq x 0) (Ex …
  -/
  rw [oangle_eq_zero_iff_sameRay, oangle_eq_pi_iff_sameRay_neg]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Or (SameRay Real x y) (And (Ne x 0) (And (Ne y 0) (SameRay Real x (Neg. …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Or (SameRay Real x y) (And (Ne x 0) (And (Ne y 0) (SameRay Real x (Neg.neg …
      ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
    -/
  · rcases h with (h | ⟨-, -, h⟩)
      /-
        case refine_1.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x y
        ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      -/
    · by_cases hx : x = 0; · simp [hx]
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x y
        hx : Not (Eq x 0)
        ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      -/
      obtain ⟨r, -, rfl⟩ := h.exists_nonneg_left hx
      /-
        case neg.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        hx : Not (Eq x 0)
        r : Real
        h : SameRay Real x (HSMul.hSMul r x)
        ⊢ Or (Eq x 0) (Exists fun r_1 => Eq (HSMul.hSMul r x) (HSMul.hSMul r_1 x))
      -/
      exact Or.inr ⟨r, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x (Neg.neg y)
        ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      -/
    · by_cases hx : x = 0; · simp [hx]
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x (Neg.neg y)
        hx : Not (Eq x 0)
        ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      -/
      obtain ⟨r, -, hy⟩ := h.exists_nonneg_left hx
      /-
        case neg.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x (Neg.neg y)
        hx : Not (Eq x 0)
        r : Real
        hy : Eq (HSMul.hSMul r x) (Neg.neg y)
        ⊢ Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      -/
      refine Or.inr ⟨-r, ?_⟩
      /-
        case neg.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        h : SameRay Real x (Neg.neg y)
        hx : Not (Eq x 0)
        r : Real
        hy : Eq (HSMul.hSMul r x) (Neg.neg y)
        ⊢ Eq y (HSMul.hSMul (Neg.neg r) x)
      -/
      simp [hy]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))
      ⊢ Or (SameRay Real x y) (And (Ne x 0) (And (Ne y 0) (SameRay Real x (Neg.neg y …
    -/
  · rcases h with (rfl | ⟨r, rfl⟩); · simp
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case refine_2.inr.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      ⊢ Or (SameRay Real x (HSMul.hSMul r x)) (And (Ne x 0) (And (Ne (HSMul.hSMul r  …
    -/
    by_cases hx : x = 0; · simp [hx]
                           /-
                             🎉 no goals
                           -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      r : Real
      hx : Not (Eq x 0)
      ⊢ Or (SameRay Real x (HSMul.hSMul r x)) (And (Ne x 0) (And (Ne (HSMul.hSMul r  …
    -/
    rcases lt_trichotomy r 0 with (hr | hr | hr)
      /-
        case neg.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        hx : Not (Eq x 0)
        hr : LT.lt r 0
        ⊢ Or (SameRay Real x (HSMul.hSMul r x)) (And (Ne x 0) (And (Ne (HSMul.hSMul r  …
      -/
    · rw [← neg_smul]
      exact Or.inr ⟨hx, smul_ne_zero hr.ne hx,
        SameRay.sameRay_pos_smul_right x (Left.neg_pos_iff.2 hr)⟩
      /-
        case neg.inr.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        hx : Not (Eq x 0)
        hr : Eq r 0
        ⊢ Or (SameRay Real x (HSMul.hSMul r x)) (And (Ne x 0) (And (Ne (HSMul.hSMul r  …
      -/
    · simp [hr]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.inr
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        hx : Not (Eq x 0)
        hr : LT.lt 0 r
        ⊢ Or (SameRay Real x (HSMul.hSMul r x)) (And (Ne x 0) (And (Ne (HSMul.hSMul r  …
      -/
    · exact Or.inl (SameRay.sameRay_pos_smul_right x hr)
      /-
        🎉 no goals
      -/


/-- The oriented angle between two vectors is not zero or `π` if and only if those two vectors
are linearly independent. -/
theorem oangle_ne_zero_and_ne_pi_iff_linearIndependent {x y : V} :
    o.oangle x y ≠ 0 ∧ o.oangle x y ≠ π ↔ LinearIndependent ℝ ![x, y] := by
  rw [← not_or, ← not_iff_not, Classical.not_not,
    oangle_eq_zero_or_eq_pi_iff_not_linearIndependent]


/-- Two vectors are equal if and only if they have equal norms and zero angle between them. -/
theorem eq_iff_norm_eq_and_oangle_eq_zero (x y : V) : x = y ↔ ‖x‖ = ‖y‖ ∧ o.oangle x y = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq x y) (And (Eq (Norm.norm x) (Norm.norm y)) (Eq (o.oangle x y) 0))
  -/
  rw [oangle_eq_zero_iff_sameRay]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq x y) (And (Eq (Norm.norm x) (Norm.norm y)) (SameRay Real x y))
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ Eq x y → And (Eq (Norm.norm x) (Norm.norm y)) (SameRay Real x y)
    -/
  · rintro rfl
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      ⊢ And (Eq (Norm.norm x) (Norm.norm x)) (SameRay Real x x)
    -/
    simp; rfl
          /-
            🎉 no goals
          -/
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ And (Eq (Norm.norm x) (Norm.norm y)) (SameRay Real x y) → Eq x y
    -/
  · rcases eq_or_ne y 0 with (rfl | hy)
      /-
        case mpr.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        ⊢ And (Eq (Norm.norm x) (Norm.norm 0)) (SameRay Real x 0) → Eq x 0
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hy : Ne y 0
      ⊢ And (Eq (Norm.norm x) (Norm.norm y)) (SameRay Real x y) → Eq x y
    -/
    rintro ⟨h₁, h₂⟩
    /-
      case mpr.inr.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hy : Ne y 0
      h₁ : Eq (Norm.norm x) (Norm.norm y)
      h₂ : SameRay Real x y
      ⊢ Eq x y
    -/
    obtain ⟨r, hr, rfl⟩ := h₂.exists_nonneg_right hy
    /-
      case mpr.inr.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      y : V
      hy : Ne y 0
      r : Real
      hr : LE.le 0 r
      h₁ : Eq (Norm.norm (HSMul.hSMul r y)) (Norm.norm y)
      h₂ : SameRay Real (HSMul.hSMul r y) y
      ⊢ Eq (HSMul.hSMul r y) y
    -/
    have : ‖y‖ ≠ 0 := by simpa using hy
    obtain rfl : r = 1 := by
      apply mul_right_cancel₀ this
      simpa [norm_smul, _root_.abs_of_nonneg hr] using h₁
    /-
      case mpr.inr.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      y : V
      hy : Ne y 0
      this : Ne (Norm.norm y) 0
      hr : LE.le 0 1
      h₁ : Eq (Norm.norm (HSMul.hSMul 1 y)) (Norm.norm y)
      h₂ : SameRay Real (HSMul.hSMul 1 y) y
      ⊢ Eq (HSMul.hSMul 1 y) y
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Two vectors with equal norms are equal if and only if they have zero angle between them. -/
theorem eq_iff_oangle_eq_zero_of_norm_eq {x y : V} (h : ‖x‖ = ‖y‖) : x = y ↔ o.oangle x y = 0 :=
  ⟨fun he => ((o.eq_iff_norm_eq_and_oangle_eq_zero x y).1 he).2, fun ha =>
    (o.eq_iff_norm_eq_and_oangle_eq_zero x y).2 ⟨h, ha⟩⟩


/-- Two vectors with zero angle between them are equal if and only if they have equal norms. -/
theorem eq_iff_norm_eq_of_oangle_eq_zero {x y : V} (h : o.oangle x y = 0) : x = y ↔ ‖x‖ = ‖y‖ :=
  ⟨fun he => ((o.eq_iff_norm_eq_and_oangle_eq_zero x y).1 he).1, fun hn =>
    (o.eq_iff_norm_eq_and_oangle_eq_zero x y).2 ⟨hn, h⟩⟩


/-- Given three nonzero vectors, the angle between the first and the second plus the angle
between the second and the third equals the angle between the first and the third. -/
@[simp]
theorem oangle_add {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    o.oangle x y + o.oangle y z = o.oangle x z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Eq (HAdd.hAdd (o.oangle x y) (o.oangle y z)) (o.oangle x z)
  -/
  simp_rw [oangle]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Eq (HAdd.hAdd ↑((o.kahler x) y).arg ↑((o.kahler y) z).arg) ↑((o.kahler x) z) …
  -/
  rw [← Complex.arg_mul_coe_angle, o.kahler_mul y x z]
    /-
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y z : V
      hx : Ne x 0
      hy : Ne y 0
      hz : Ne z 0
      ⊢ Eq ↑(HMul.hMul (HPow.hPow (↑(Norm.norm y)) 2) ((o.kahler x) z)).arg ↑((o.kah …
    -/
  · congr 1
    /-
      case e_r
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y z : V
      hx : Ne x 0
      hy : Ne y 0
      hz : Ne z 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Norm.norm y)) 2) ((o.kahler x) z)).arg ((o.kahle …
    -/
    exact mod_cast Complex.arg_real_mul _ (by positivity : 0 < ‖y‖ ^ 2)
    /-
      🎉 no goals
    -/
    /-
      case hx
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y z : V
      hx : Ne x 0
      hy : Ne y 0
      hz : Ne z 0
      ⊢ Ne ((o.kahler x) y) 0
    -/
  · exact o.kahler_ne_zero hx hy
    /-
      🎉 no goals
    -/
    /-
      case hy
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y z : V
      hx : Ne x 0
      hy : Ne y 0
      hz : Ne z 0
      ⊢ Ne ((o.kahler y) z) 0
    -/
  · exact o.kahler_ne_zero hy hz
    /-
      🎉 no goals
    -/


/-- Given three nonzero vectors, the angle between the second and the third plus the angle
between the first and the second equals the angle between the first and the third. -/
@[simp]
theorem oangle_add_swap {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
                                                     /-
                                                       V : Type u_1
                                                       inst✝² : NormedAddCommGroup V
                                                       inst✝¹ : InnerProductSpace Real V
                                                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                       o : Orientation Real V (Fin 2)
                                                       x y z : V
                                                       hx : Ne x 0
                                                       hy : Ne y 0
                                                       hz : Ne z 0
                                                       ⊢ Eq (HAdd.hAdd (o.oangle y z) (o.oangle x y)) (o.oangle x z)
                                                     -/
    o.oangle y z + o.oangle x y = o.oangle x z := by rw [add_comm, o.oangle_add hx hy hz]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Given three nonzero vectors, the angle between the first and the third minus the angle
between the first and the second equals the angle between the second and the third. -/
@[simp]
theorem oangle_sub_left {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    o.oangle x z - o.oangle x y = o.oangle y z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Eq (HSub.hSub (o.oangle x z) (o.oangle x y)) (o.oangle y z)
  -/
  rw [sub_eq_iff_eq_add, o.oangle_add_swap hx hy hz]
  /-
    🎉 no goals
  -/


/-- Given three nonzero vectors, the angle between the first and the third minus the angle
between the second and the third equals the angle between the first and the second. -/
@[simp]
theorem oangle_sub_right {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
                                                     /-
                                                       V : Type u_1
                                                       inst✝² : NormedAddCommGroup V
                                                       inst✝¹ : InnerProductSpace Real V
                                                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                       o : Orientation Real V (Fin 2)
                                                       x y z : V
                                                       hx : Ne x 0
                                                       hy : Ne y 0
                                                       hz : Ne z 0
                                                       ⊢ Eq (HSub.hSub (o.oangle x z) (o.oangle y z)) (o.oangle x y)
                                                     -/
    o.oangle x z - o.oangle y z = o.oangle x y := by rw [sub_eq_iff_eq_add, o.oangle_add hx hy hz]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Given three nonzero vectors, adding the angles between them in cyclic order results in 0. -/
@[simp]
theorem oangle_add_cyc3 {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
                                                         /-
                                                           V : Type u_1
                                                           inst✝² : NormedAddCommGroup V
                                                           inst✝¹ : InnerProductSpace Real V
                                                           inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                           o : Orientation Real V (Fin 2)
                                                           x y z : V
                                                           hx : Ne x 0
                                                           hy : Ne y 0
                                                           hz : Ne z 0
                                                           ⊢ Eq (HAdd.hAdd (HAdd.hAdd (o.oangle x y) (o.oangle y z)) (o.oangle z x)) 0
                                                         -/
    o.oangle x y + o.oangle y z + o.oangle z x = 0 := by simp [hx, hy, hz]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Given three nonzero vectors, adding the angles between them in cyclic order, with the first
vector in each angle negated, results in π. If the vectors add to 0, this is a version of the
sum of the angles of a triangle. -/
@[simp]
theorem oangle_add_cyc3_neg_left {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    o.oangle (-x) y + o.oangle (-y) z + o.oangle (-z) x = π := by
  rw [o.oangle_neg_left hx hy, o.oangle_neg_left hy hz, o.oangle_neg_left hz hx,
    show o.oangle x y + π + (o.oangle y z + π) + (o.oangle z x + π) =
      o.oangle x y + o.oangle y z + o.oangle z x + (π + π + π : Real.Angle) by abel,
    o.oangle_add_cyc3 hx hy hz, Real.Angle.coe_pi_add_coe_pi, zero_add, zero_add]


/-- Given three nonzero vectors, adding the angles between them in cyclic order, with the second
vector in each angle negated, results in π. If the vectors add to 0, this is a version of the
sum of the angles of a triangle. -/
@[simp]
theorem oangle_add_cyc3_neg_right {x y z : V} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    o.oangle x (-y) + o.oangle y (-z) + o.oangle z (-x) = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y z : V
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (o.oangle x (Neg.neg y)) (o.oangle y (Neg.neg z)))  …
  -/
  simp_rw [← oangle_neg_left_eq_neg_right, o.oangle_add_cyc3_neg_left hx hy hz]
  /-
    🎉 no goals
  -/


/-- Pons asinorum, oriented vector angle form. -/
theorem oangle_sub_eq_oangle_sub_rev_of_norm_eq {x y : V} (h : ‖x‖ = ‖y‖) :
                                                  /-
                                                    V : Type u_1
                                                    inst✝² : NormedAddCommGroup V
                                                    inst✝¹ : InnerProductSpace Real V
                                                    inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                    o : Orientation Real V (Fin 2)
                                                    x y : V
                                                    h : Eq (Norm.norm x) (Norm.norm y)
                                                    ⊢ Eq (o.oangle x (HSub.hSub x y)) (o.oangle (HSub.hSub y x) y)
                                                  -/
    o.oangle x (x - y) = o.oangle (y - x) y := by simp [oangle, h]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The angle at the apex of an isosceles triangle is `π` minus twice a base angle, oriented
vector angle form. -/
theorem oangle_eq_pi_sub_two_zsmul_oangle_sub_of_norm_eq {x y : V} (hn : x ≠ y) (h : ‖x‖ = ‖y‖) :
    o.oangle y x = π - (2 : ℤ) • o.oangle (y - x) y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ Eq (o.oangle y x) (HSub.hSub (↑Real.pi) (HSMul.hSMul 2 (o.oangle (HSub.hSub  …
  -/
  rw [two_zsmul]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ Eq (o.oangle y x) (HSub.hSub (↑Real.pi) (HAdd.hAdd (o.oangle (HSub.hSub y x) …
  -/
  nth_rw 1 [← o.oangle_sub_eq_oangle_sub_rev_of_norm_eq h]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ Eq (o.oangle y x) (HSub.hSub (↑Real.pi) (HAdd.hAdd (o.oangle x (HSub.hSub x  …
  -/
  rw [eq_sub_iff_add_eq, ← oangle_neg_neg, ← add_assoc]
  have hy : y ≠ 0 := by
    rintro rfl
    rw [norm_zero, norm_eq_zero] at h
    exact hn h
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (o.oangle (Neg.neg y) (Neg.neg x)) (o.oangle x (HSu …
  -/
  have hx : x ≠ 0 := norm_ne_zero_iff.1 (h.symm ▸ norm_ne_zero_iff.2 hy)
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    hy : Ne y 0
    hx : Ne x 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (o.oangle (Neg.neg y) (Neg.neg x)) (o.oangle x (HSu …
  -/
  convert o.oangle_add_cyc3_neg_right (neg_ne_zero.2 hy) hx (sub_ne_zero_of_ne hn.symm) using 1
  /-
    case h.e'_2
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hn : Ne x y
    h : Eq (Norm.norm x) (Norm.norm y)
    hy : Ne y 0
    hx : Ne x 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (o.oangle (Neg.neg y) (Neg.neg x)) (o.oangle x (HSu …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The angle between two vectors, with respect to an orientation given by `Orientation.map`
with a linear isometric equivalence, equals the angle between those two vectors, transformed by
the inverse of that equivalence, with respect to the original orientation. -/
@[simp]
theorem oangle_map (x y : V') (f : V ≃ₗᵢ[ℝ] V') :
    (Orientation.map (Fin 2) f.toLinearEquiv o).oangle x y = o.oangle (f.symm x) (f.symm y) := by
  /-
    V : Type u_1
    V' : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedAddCommGroup V'
    inst✝³ : InnerProductSpace Real V
    inst✝² : InnerProductSpace Real V'
    inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Fact (Eq (Module.finrank Real V') 2)
    o : Orientation Real V (Fin 2)
    x y : V'
    f : LinearIsometryEquiv (RingHom.id Real) V V'
    ⊢ Eq (((Orientation.map (Fin 2) f.toLinearEquiv) o).oangle x y) (o.oangle (f.s …
  -/
  simp [oangle, o.kahler_map]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem _root_.Complex.oangle (w z : ℂ) :
                                                                    /-
                                                                      w z : Complex
                                                                      ⊢ Eq (Complex.orientation.oangle w z) ↑(HMul.hMul ((starRingEnd Complex) w) z) …
                                                                    -/
    Complex.orientation.oangle w z = Complex.arg (conj w * z) := by simp [oangle]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The oriented angle on an oriented real inner product space of dimension 2 can be evaluated in
terms of a complex-number representation of the space. -/
theorem oangle_map_complex (f : V ≃ₗᵢ[ℝ] ℂ)
    (hf : Orientation.map (Fin 2) f.toLinearEquiv o = Complex.orientation) (x y : V) :
    o.oangle x y = Complex.arg (conj (f x) * f y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : V
    ⊢ Eq (o.oangle x y) ↑(HMul.hMul ((starRingEnd Complex) (f x)) (f y)).arg
  -/
  rw [← Complex.oangle, ← hf, o.oangle_map]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x y : V
    ⊢ Eq (o.oangle x y) (o.oangle (f.symm (f x)) (f.symm (f y)))
  -/
  iterate 2 rw [LinearIsometryEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Negating the orientation negates the value of `oangle`. -/
theorem oangle_neg_orientation_eq_neg (x y : V) : (-o).oangle x y = -o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq ((Neg.neg o).oangle x y) (Neg.neg (o.oangle x y))
  -/
  simp [oangle]
  /-
    🎉 no goals
  -/


/-- The inner product of two vectors is the product of the norms and the cosine of the oriented
angle between the vectors. -/
theorem inner_eq_norm_mul_norm_mul_cos_oangle (x y : V) :
    ⟪x, y⟫ = ‖x‖ * ‖y‖ * Real.Angle.cos (o.oangle x y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (Inner.inner x y) (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.norm y)) (o.o …
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    ⊢ Eq (Inner.inner x y) (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.norm y)) (o.o …
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (Inner.inner x y) (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.norm y)) (o.o …
  -/
  rw [oangle, Real.Angle.cos_coe, Complex.cos_arg, o.abs_kahler]
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      ⊢ Eq (Inner.inner x y) (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.norm y)) (HDi …
    -/
  · simp only [kahler_apply_apply, real_smul, add_re, ofReal_re, mul_re, I_re, ofReal_im]
    -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp`; replaced by `simp only ...` to speed up
    -- Reinstate `field_simp` once it is faster.
    simp (disch := field_simp_discharge) only [mul_zero, I_im, mul_one, sub_self, add_zero,
      mul_div_assoc', mul_div_cancel_left₀]
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      ⊢ Ne ((o.kahler x) y) 0
    -/
  · exact o.kahler_ne_zero hx hy
    /-
      🎉 no goals
    -/


/-- The cosine of the oriented angle between two nonzero vectors is the inner product divided by
the product of the norms. -/
theorem cos_oangle_eq_inner_div_norm_mul_norm {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.Angle.cos (o.oangle x y) = ⟪x, y⟫ / (‖x‖ * ‖y‖) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (o.oangle x y).cos (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x)  …
  -/
  rw [o.inner_eq_norm_mul_norm_mul_cos_oangle]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (o.oangle x y).cos (HDiv.hDiv (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.n …
  -/
  field_simp [norm_ne_zero_iff.2 hx, norm_ne_zero_iff.2 hy]
  /-
    🎉 no goals
  -/


/-- The cosine of the oriented angle between two nonzero vectors equals that of the unoriented
angle. -/
theorem cos_oangle_eq_cos_angle {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.Angle.cos (o.oangle x y) = Real.cos (InnerProductGeometry.angle x y) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (o.oangle x y).cos (Real.cos (InnerProductGeometry.angle x y))
  -/
  rw [o.cos_oangle_eq_inner_div_norm_mul_norm hx hy, InnerProductGeometry.cos_angle]
  /-
    🎉 no goals
  -/


/-- The oriented angle between two nonzero vectors is plus or minus the unoriented angle. -/
theorem oangle_eq_angle_or_eq_neg_angle {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    o.oangle x y = InnerProductGeometry.angle x y ∨
      o.oangle x y = -InnerProductGeometry.angle x y :=
  Real.Angle.cos_eq_real_cos_iff_eq_or_eq_neg.1 <| o.cos_oangle_eq_cos_angle hx hy


/-- The unoriented angle between two nonzero vectors is the absolute value of the oriented angle,
converted to a real. -/
theorem angle_eq_abs_oangle_toReal {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    InnerProductGeometry.angle x y = |(o.oangle x y).toReal| := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (InnerProductGeometry.angle x y) (abs (o.oangle x y).toReal)
  -/
  have h0 := InnerProductGeometry.angle_nonneg x y
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h0 : LE.le 0 (InnerProductGeometry.angle x y)
    ⊢ Eq (InnerProductGeometry.angle x y) (abs (o.oangle x y).toReal)
  -/
  have hpi := InnerProductGeometry.angle_le_pi x y
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h0 : LE.le 0 (InnerProductGeometry.angle x y)
    hpi : LE.le (InnerProductGeometry.angle x y) Real.pi
    ⊢ Eq (InnerProductGeometry.angle x y) (abs (o.oangle x y).toReal)
  -/
  rcases o.oangle_eq_angle_or_eq_neg_angle hx hy with (h | h)
    /-
      case inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h0 : LE.le 0 (InnerProductGeometry.angle x y)
      hpi : LE.le (InnerProductGeometry.angle x y) Real.pi
      h : Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
      ⊢ Eq (InnerProductGeometry.angle x y) (abs (o.oangle x y).toReal)
    -/
  · rw [h, eq_comm, Real.Angle.abs_toReal_coe_eq_self_iff]
    /-
      case inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h0 : LE.le 0 (InnerProductGeometry.angle x y)
      hpi : LE.le (InnerProductGeometry.angle x y) Real.pi
      h : Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
      ⊢ And (LE.le 0 (InnerProductGeometry.angle x y)) (LE.le (InnerProductGeometry. …
    -/
    exact ⟨h0, hpi⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h0 : LE.le 0 (InnerProductGeometry.angle x y)
      hpi : LE.le (InnerProductGeometry.angle x y) Real.pi
      h : Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
      ⊢ Eq (InnerProductGeometry.angle x y) (abs (o.oangle x y).toReal)
    -/
  · rw [h, eq_comm, Real.Angle.abs_toReal_neg_coe_eq_self_iff]
    /-
      case inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h0 : LE.le 0 (InnerProductGeometry.angle x y)
      hpi : LE.le (InnerProductGeometry.angle x y) Real.pi
      h : Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
      ⊢ And (LE.le 0 (InnerProductGeometry.angle x y)) (LE.le (InnerProductGeometry. …
    -/
    exact ⟨h0, hpi⟩
    /-
      🎉 no goals
    -/


/-- If the sign of the oriented angle between two vectors is zero, either one of the vectors is
zero or the unoriented angle is 0 or π. -/
theorem eq_zero_or_angle_eq_zero_or_pi_of_sign_oangle_eq_zero {x y : V}
    (h : (o.oangle x y).sign = 0) :
    x = 0 ∨ y = 0 ∨ InnerProductGeometry.angle x y = 0 ∨ InnerProductGeometry.angle x y = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 0
    ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (InnerProductGeometry.angle x y) 0) (Eq (In …
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 0
    hx : Not (Eq x 0)
    ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (InnerProductGeometry.angle x y) 0) (Eq (In …
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 0
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (InnerProductGeometry.angle x y) 0) (Eq (In …
  -/
  rw [o.angle_eq_abs_oangle_toReal hx hy]
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 0
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (abs (o.oangle x y).toReal) 0) (Eq (abs (o. …
  -/
  rw [Real.Angle.sign_eq_zero_iff] at h
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Or (Eq (o.oangle x y) 0) (Eq (o.oangle x y) ↑Real.pi)
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (abs (o.oangle x y).toReal) 0) (Eq (abs (o. …
  -/
                            /-
                              🎉 no goals
                            -/
  rcases h with (h | h) <;> simp [h, Real.pi_pos.le]
                            /-
                              🎉 no goals
                            -/


/-- If two unoriented angles are equal, and the signs of the corresponding oriented angles are
equal, then the oriented angles are equal (even in degenerate cases). -/
theorem oangle_eq_of_angle_eq_of_sign_eq {w x y z : V}
    (h : InnerProductGeometry.angle w x = InnerProductGeometry.angle y z)
    (hs : (o.oangle w x).sign = (o.oangle y z).sign) : o.oangle w x = o.oangle y z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    w x y z : V
    h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
    hs : Eq (o.oangle w x).sign (o.oangle y z).sign
    ⊢ Eq (o.oangle w x) (o.oangle y z)
  -/
  by_cases h0 : (w = 0 ∨ x = 0) ∨ y = 0 ∨ z = 0
  · have hs' : (o.oangle w x).sign = 0 ∧ (o.oangle y z).sign = 0 := by
      rcases h0 with ((rfl | rfl) | rfl | rfl)
      · simpa using hs.symm
      · simpa using hs.symm
      · simpa using hs
      · simpa using hs
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      w x y z : V
      h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
      hs : Eq (o.oangle w x).sign (o.oangle y z).sign
      h0 : Or (Or (Eq w 0) (Eq x 0)) (Or (Eq y 0) (Eq z 0))
      hs' : And (Eq (o.oangle w x).sign 0) (Eq (o.oangle y z).sign 0)
      ⊢ Eq (o.oangle w x) (o.oangle y z)
    -/
    rcases hs' with ⟨hswx, hsyz⟩
    have h' : InnerProductGeometry.angle w x = π / 2 ∧ InnerProductGeometry.angle y z = π / 2 := by
      rcases h0 with ((rfl | rfl) | rfl | rfl)
      · simpa using h.symm
      · simpa using h.symm
      · simpa using h
      · simpa using h
    /-
      case pos.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      w x y z : V
      h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
      hs : Eq (o.oangle w x).sign (o.oangle y z).sign
      h0 : Or (Or (Eq w 0) (Eq x 0)) (Or (Eq y 0) (Eq z 0))
      hswx : Eq (o.oangle w x).sign 0
      hsyz : Eq (o.oangle y z).sign 0
      h' : And (Eq (InnerProductGeometry.angle w x) (HDiv.hDiv Real.pi 2)) (Eq (Inne …
      ⊢ Eq (o.oangle w x) (o.oangle y z)
    -/
    rcases h' with ⟨hwx, hyz⟩
    have hpi : π / 2 ≠ π := by
      intro hpi
      rw [div_eq_iff, eq_comm, ← sub_eq_zero, mul_two, add_sub_cancel_right] at hpi
      · exact Real.pi_pos.ne.symm hpi
      · exact two_ne_zero
    have h0wx : w = 0 ∨ x = 0 := by
      have h0' := o.eq_zero_or_angle_eq_zero_or_pi_of_sign_oangle_eq_zero hswx
      simpa [hwx, Real.pi_pos.ne.symm, hpi] using h0'
    have h0yz : y = 0 ∨ z = 0 := by
      have h0' := o.eq_zero_or_angle_eq_zero_or_pi_of_sign_oangle_eq_zero hsyz
      simpa [hyz, Real.pi_pos.ne.symm, hpi] using h0'
    /-
      case pos.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      w x y z : V
      h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
      hs : Eq (o.oangle w x).sign (o.oangle y z).sign
      h0 : Or (Or (Eq w 0) (Eq x 0)) (Or (Eq y 0) (Eq z 0))
      hswx : Eq (o.oangle w x).sign 0
      hsyz : Eq (o.oangle y z).sign 0
      hwx : Eq (InnerProductGeometry.angle w x) (HDiv.hDiv Real.pi 2)
      hyz : Eq (InnerProductGeometry.angle y z) (HDiv.hDiv Real.pi 2)
      hpi : Ne (HDiv.hDiv Real.pi 2) Real.pi
      h0wx : Or (Eq w 0) (Eq x 0)
      h0yz : Or (Eq y 0) (Eq z 0)
      ⊢ Eq (o.oangle w x) (o.oangle y z)
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
    rcases h0wx with (h0wx | h0wx) <;> rcases h0yz with (h0yz | h0yz) <;> simp [h0wx, h0yz]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      w x y z : V
      h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
      hs : Eq (o.oangle w x).sign (o.oangle y z).sign
      h0 : Not (Or (Or (Eq w 0) (Eq x 0)) (Or (Eq y 0) (Eq z 0)))
      ⊢ Eq (o.oangle w x) (o.oangle y z)
    -/
  · push_neg at h0
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      w x y z : V
      h : Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
      hs : Eq (o.oangle w x).sign (o.oangle y z).sign
      h0 : And (And (Ne w 0) (Ne x 0)) (And (Ne y 0) (Ne z 0))
      ⊢ Eq (o.oangle w x) (o.oangle y z)
    -/
    rw [Real.Angle.eq_iff_abs_toReal_eq_of_sign_eq hs]
    rwa [o.angle_eq_abs_oangle_toReal h0.1.1 h0.1.2,
      o.angle_eq_abs_oangle_toReal h0.2.1 h0.2.2] at h


/-- If the signs of two oriented angles between nonzero vectors are equal, the oriented angles are
equal if and only if the unoriented angles are equal. -/
theorem angle_eq_iff_oangle_eq_of_sign_eq {w x y z : V} (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0)
    (hz : z ≠ 0) (hs : (o.oangle w x).sign = (o.oangle y z).sign) :
    InnerProductGeometry.angle w x = InnerProductGeometry.angle y z ↔
    o.oangle w x = o.oangle y z := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    w x y z : V
    hw : Ne w 0
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    hs : Eq (o.oangle w x).sign (o.oangle y z).sign
    ⊢ Iff (Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)) ( …
  -/
  refine ⟨fun h => o.oangle_eq_of_angle_eq_of_sign_eq h hs, fun h => ?_⟩
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    w x y z : V
    hw : Ne w 0
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    hs : Eq (o.oangle w x).sign (o.oangle y z).sign
    h : Eq (o.oangle w x) (o.oangle y z)
    ⊢ Eq (InnerProductGeometry.angle w x) (InnerProductGeometry.angle y z)
  -/
  rw [o.angle_eq_abs_oangle_toReal hw hx, o.angle_eq_abs_oangle_toReal hy hz, h]
  /-
    🎉 no goals
  -/


/-- The oriented angle between two vectors equals the unoriented angle if the sign is positive. -/
theorem oangle_eq_angle_of_sign_eq_one {x y : V} (h : (o.oangle x y).sign = 1) :
    o.oangle x y = InnerProductGeometry.angle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 1
    ⊢ Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
  -/
  by_cases hx : x = 0; · exfalso; simp [hx] at h
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 1
    hx : Not (Eq x 0)
    ⊢ Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
  -/
  by_cases hy : y = 0; · exfalso; simp [hy] at h
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 1
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
  -/
  refine (o.oangle_eq_angle_or_eq_neg_angle hx hy).resolve_right ?_
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 1
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Not (Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y)))
  -/
  intro hxy
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign 1
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hxy : Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
    ⊢ False
  -/
  rw [hxy, Real.Angle.sign_neg, neg_eq_iff_eq_neg, ← SignType.neg_iff, ← not_le] at h
  exact h (Real.Angle.sign_coe_nonneg_of_nonneg_of_le_pi (InnerProductGeometry.angle_nonneg _ _)
    (InnerProductGeometry.angle_le_pi _ _))


/-- The oriented angle between two vectors equals minus the unoriented angle if the sign is
negative. -/
theorem oangle_eq_neg_angle_of_sign_eq_neg_one {x y : V} (h : (o.oangle x y).sign = -1) :
    o.oangle x y = -InnerProductGeometry.angle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign (-1)
    ⊢ Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
  -/
  by_cases hx : x = 0; · exfalso; simp [hx] at h
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign (-1)
    hx : Not (Eq x 0)
    ⊢ Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
  -/
  by_cases hy : y = 0; · exfalso; simp [hy] at h
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign (-1)
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (o.oangle x y) (Neg.neg ↑(InnerProductGeometry.angle x y))
  -/
  refine (o.oangle_eq_angle_or_eq_neg_angle hx hy).resolve_left ?_
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign (-1)
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Not (Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y))
  -/
  intro hxy
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (o.oangle x y).sign (-1)
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hxy : Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)
    ⊢ False
  -/
  rw [hxy, ← SignType.neg_iff, ← not_le] at h
  exact h (Real.Angle.sign_coe_nonneg_of_nonneg_of_le_pi (InnerProductGeometry.angle_nonneg _ _)
    (InnerProductGeometry.angle_le_pi _ _))


/-- The oriented angle between two nonzero vectors is zero if and only if the unoriented angle
is zero. -/
theorem oangle_eq_zero_iff_angle_eq_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    o.oangle x y = 0 ↔ InnerProductGeometry.angle x y = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (o.oangle x y) 0) (Eq (InnerProductGeometry.angle x y) 0)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (o.oangle x y) 0
      ⊢ Eq (InnerProductGeometry.angle x y) 0
    -/
  · simpa [o.angle_eq_abs_oangle_toReal hx hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (InnerProductGeometry.angle x y) 0
      ⊢ Eq (o.oangle x y) 0
    -/
  · have ha := o.oangle_eq_angle_or_eq_neg_angle hx hy
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (InnerProductGeometry.angle x y) 0
      ha : Or (Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)) (Eq (o.oangle x  …
      ⊢ Eq (o.oangle x y) 0
    -/
    rw [h] at ha
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      h : Eq (InnerProductGeometry.angle x y) 0
      ha : Or (Eq (o.oangle x y) ↑0) (Eq (o.oangle x y) (Neg.neg ↑0))
      ⊢ Eq (o.oangle x y) 0
    -/
    simpa using ha
    /-
      🎉 no goals
    -/


/-- The oriented angle between two vectors is `π` if and only if the unoriented angle is `π`. -/
theorem oangle_eq_pi_iff_angle_eq_pi {x y : V} :
    o.oangle x y = π ↔ InnerProductGeometry.angle x y = π := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (Eq (InnerProductGeometry.angle x y) Real.pi)
  -/
  by_cases hx : x = 0
  · simp [hx, Real.Angle.pi_ne_zero.symm, div_eq_mul_inv, mul_right_eq_self₀, not_or,
      Real.pi_ne_zero]
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (Eq (InnerProductGeometry.angle x y) Real.pi)
  -/
  by_cases hy : y = 0
  · simp [hy, Real.Angle.pi_ne_zero.symm, div_eq_mul_inv, mul_right_eq_self₀, not_or,
      Real.pi_ne_zero]
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Eq (o.oangle x y) ↑Real.pi) (Eq (InnerProductGeometry.angle x y) Real.pi)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case neg.refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (o.oangle x y) ↑Real.pi
      ⊢ Eq (InnerProductGeometry.angle x y) Real.pi
    -/
  · rw [o.angle_eq_abs_oangle_toReal hx hy, h]
    /-
      case neg.refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (o.oangle x y) ↑Real.pi
      ⊢ Eq (abs (↑Real.pi).toReal) Real.pi
    -/
    simp [Real.pi_pos.le]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (InnerProductGeometry.angle x y) Real.pi
      ⊢ Eq (o.oangle x y) ↑Real.pi
    -/
  · have ha := o.oangle_eq_angle_or_eq_neg_angle hx hy
    /-
      case neg.refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (InnerProductGeometry.angle x y) Real.pi
      ha : Or (Eq (o.oangle x y) ↑(InnerProductGeometry.angle x y)) (Eq (o.oangle x  …
      ⊢ Eq (o.oangle x y) ↑Real.pi
    -/
    rw [h] at ha
    /-
      case neg.refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (InnerProductGeometry.angle x y) Real.pi
      ha : Or (Eq (o.oangle x y) ↑Real.pi) (Eq (o.oangle x y) (Neg.neg ↑Real.pi))
      ⊢ Eq (o.oangle x y) ↑Real.pi
    -/
    simpa using ha
    /-
      🎉 no goals
    -/


/-- One of two vectors is zero or the oriented angle between them is plus or minus `π / 2` if
and only if the inner product of those vectors is zero. -/
theorem eq_zero_or_oangle_eq_iff_inner_eq_zero {x y : V} :
    x = 0 ∨ y = 0 ∨ o.oangle x y = (π / 2 : ℝ) ∨ o.oangle x y = (-π / 2 : ℝ) ↔ ⟪x, y⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) …
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) …
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) …
  -/
  rw [InnerProductGeometry.inner_eq_zero_iff_angle_eq_pi_div_two, or_iff_right hx, or_iff_right hy]
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) (Eq (o.oangle x y) ↑(HDiv …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case neg.refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) (Eq (o.oangle x y) ↑(HDiv.hD …
      ⊢ Eq (InnerProductGeometry.angle x y) (HDiv.hDiv Real.pi 2)
    -/
  · rwa [o.angle_eq_abs_oangle_toReal hx hy, Real.Angle.abs_toReal_eq_pi_div_two_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (InnerProductGeometry.angle x y) (HDiv.hDiv Real.pi 2)
      ⊢ Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) (Eq (o.oangle x y) ↑(HDiv.hDiv …
    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  · convert o.oangle_eq_angle_or_eq_neg_angle hx hy using 2 <;> rw [h]
    /-
      case h.e'_2.h.e'_3
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Eq (InnerProductGeometry.angle x y) (HDiv.hDiv Real.pi 2)
      ⊢ Eq (↑(HDiv.hDiv (Neg.neg Real.pi) 2)) (Neg.neg ↑(HDiv.hDiv Real.pi 2))
    -/
    simp only [neg_div, Real.Angle.coe_neg]
    /-
      🎉 no goals
    -/


/-- If the oriented angle between two vectors is `π / 2`, the inner product of those vectors
is zero. -/
theorem inner_eq_zero_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = (π / 2 : ℝ)) :
    ⟪x, y⟫ = 0 :=
  o.eq_zero_or_oangle_eq_iff_inner_eq_zero.1 <| Or.inr <| Or.inr <| Or.inl h


/-- If the oriented angle between two vectors is `π / 2`, the inner product of those vectors
(reversed) is zero. -/
theorem inner_rev_eq_zero_of_oangle_eq_pi_div_two {x y : V} (h : o.oangle x y = (π / 2 : ℝ)) :
                     /-
                       V : Type u_1
                       inst✝² : NormedAddCommGroup V
                       inst✝¹ : InnerProductSpace Real V
                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                       o : Orientation Real V (Fin 2)
                       x y : V
                       h : Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)
                       ⊢ Eq (Inner.inner y x) 0
                     -/
    ⟪y, x⟫ = 0 := by rw [real_inner_comm, o.inner_eq_zero_of_oangle_eq_pi_div_two h]
                     /-
                       🎉 no goals
                     -/


/-- If the oriented angle between two vectors is `-π / 2`, the inner product of those vectors
is zero. -/
theorem inner_eq_zero_of_oangle_eq_neg_pi_div_two {x y : V} (h : o.oangle x y = (-π / 2 : ℝ)) :
    ⟪x, y⟫ = 0 :=
  o.eq_zero_or_oangle_eq_iff_inner_eq_zero.1 <| Or.inr <| Or.inr <| Or.inr h


/-- If the oriented angle between two vectors is `-π / 2`, the inner product of those vectors
(reversed) is zero. -/
theorem inner_rev_eq_zero_of_oangle_eq_neg_pi_div_two {x y : V} (h : o.oangle x y = (-π / 2 : ℝ)) :
                     /-
                       V : Type u_1
                       inst✝² : NormedAddCommGroup V
                       inst✝¹ : InnerProductSpace Real V
                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                       o : Orientation Real V (Fin 2)
                       x y : V
                       h : Eq (o.oangle x y) ↑(HDiv.hDiv (Neg.neg Real.pi) 2)
                       ⊢ Eq (Inner.inner y x) 0
                     -/
    ⟪y, x⟫ = 0 := by rw [real_inner_comm, o.inner_eq_zero_of_oangle_eq_neg_pi_div_two h]
                     /-
                       🎉 no goals
                     -/


/-- Negating the first vector passed to `oangle` negates the sign of the angle. -/
@[simp]
theorem oangle_sign_neg_left (x y : V) : (o.oangle (-x) y).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (Neg.neg x) y).sign (Neg.neg (o.oangle x y).sign)
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    ⊢ Eq (o.oangle (Neg.neg x) y).sign (Neg.neg (o.oangle x y).sign)
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (o.oangle (Neg.neg x) y).sign (Neg.neg (o.oangle x y).sign)
  -/
  rw [o.oangle_neg_left hx hy, Real.Angle.sign_add_pi]
  /-
    🎉 no goals
  -/


/-- Negating the second vector passed to `oangle` negates the sign of the angle. -/
@[simp]
theorem oangle_sign_neg_right (x y : V) : (o.oangle x (-y)).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle x (Neg.neg y)).sign (Neg.neg (o.oangle x y).sign)
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    ⊢ Eq (o.oangle x (Neg.neg y)).sign (Neg.neg (o.oangle x y).sign)
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (o.oangle x (Neg.neg y)).sign (Neg.neg (o.oangle x y).sign)
  -/
  rw [o.oangle_neg_right hx hy, Real.Angle.sign_add_pi]
  /-
    🎉 no goals
  -/


/-- Multiplying the first vector passed to `oangle` by a real multiplies the sign of the angle by
the sign of the real. -/
@[simp]
theorem oangle_sign_smul_left (x y : V) (r : ℝ) :
    (o.oangle (r • x) y).sign = SignType.sign r * (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle (HSMul.hSMul r x) y).sign (HMul.hMul (SignType.sign r) (o.oangl …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  rcases lt_trichotomy r 0 with (h | h | h) <;> simp [h]
                                                /-
                                                  🎉 no goals
                                                -/


/-- Multiplying the second vector passed to `oangle` by a real multiplies the sign of the angle by
the sign of the real. -/
@[simp]
theorem oangle_sign_smul_right (x y : V) (r : ℝ) :
    (o.oangle x (r • y)).sign = SignType.sign r * (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle x (HSMul.hSMul r y)).sign (HMul.hMul (SignType.sign r) (o.oangl …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  rcases lt_trichotomy r 0 with (h | h | h) <;> simp [h]
                                                /-
                                                  🎉 no goals
                                                -/


/-- Auxiliary lemma for the proof of `oangle_sign_smul_add_right`; not intended to be used
outside of that proof. -/
theorem oangle_smul_add_right_eq_zero_or_eq_pi_iff {x y : V} (r : ℝ) :
    o.oangle x (r • x + y) = 0 ∨ o.oangle x (r • x + y) = π ↔
    o.oangle x y = 0 ∨ o.oangle x y = π := by
  simp_rw [oangle_eq_zero_or_eq_pi_iff_not_linearIndependent, Fintype.not_linearIndependent_iff,
      Fin.sum_univ_two, Fin.exists_fin_two]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Iff (Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      h : Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (M …
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
  · rcases h with ⟨m, h, hm⟩
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) (Matrix.vecCons x (Matrix.vecCons (HAdd.h …
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
    change m 0 • x + m 1 • (r • x + y) = 0 at h
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) (HAdd.hAdd (HSMul.h …
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
    refine ⟨![m 0 + m 1 * r, m 1], ?_⟩
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) (HAdd.hAdd (HSMul.h …
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (Matrix.vecCons (HAdd.hAdd (m 0) (HMul.hMul  …
    -/
    change (m 0 + m 1 * r) • x + m 1 • y = 0 ∧ (m 0 + m 1 * r ≠ 0 ∨ m 1 ≠ 0)
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) (HAdd.hAdd (HSMul.h …
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HS …
    -/
    rw [smul_add, smul_smul, ← add_assoc, ← add_smul] at h
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HSMul …
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HS …
    -/
    refine ⟨h, not_and_or.1 fun h0 => ?_⟩
    /-
      case refine_1.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HSMul …
      h0 : And (Eq (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) 0) (Eq (m 1) 0)
      ⊢ False
    -/
    obtain ⟨h0, h1⟩ := h0
    /-
      case refine_1.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HSMul …
      h0 : Eq (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    rw [h1] at h0 hm
    /-
      case refine_1.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne 0 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HSMul …
      h0 : Eq (HAdd.hAdd (m 0) (HMul.hMul 0 r)) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    rw [zero_mul, add_zero] at h0
    /-
      case refine_1.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne 0 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (m 0) (HMul.hMul (m 1) r)) x) (HSMul …
      h0 : Eq (m 0) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    simp [h0] at hm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      h : Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (M …
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
  · rcases h with ⟨m, h, hm⟩
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) (Matrix.vecCons x (Matrix.vecCons y Matri …
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
    change m 0 • x + m 1 • y = 0 at h
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      ⊢ Exists fun g => And (Eq (HAdd.hAdd (HSMul.hSMul (g 0) (Matrix.vecCons x (Mat …
    -/
    refine ⟨![m 0 - m 1 * r, m 1], ?_⟩
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (Matrix.vecCons (HSub.hSub (m 0) (HMul.hMul  …
    -/
    change (m 0 - m 1 * r) • x + m 1 • (r • x + y) = 0 ∧ (m 0 - m 1 * r ≠ 0 ∨ m 1 ≠ 0)
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub (m 0) (HMul.hMul (m 1) r)) x) (HS …
    -/
    rw [sub_smul, smul_add, smul_smul, ← add_assoc, sub_add_cancel]
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0) (Or (Ne ( …
    -/
    refine ⟨h, not_and_or.1 fun h0 => ?_⟩
    /-
      case refine_2.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      h0 : And (Eq (HSub.hSub (m 0) (HMul.hMul (m 1) r)) 0) (Eq (m 1) 0)
      ⊢ False
    -/
    obtain ⟨h0, h1⟩ := h0
    /-
      case refine_2.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne (m 1) 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      h0 : Eq (HSub.hSub (m 0) (HMul.hMul (m 1) r)) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    rw [h1] at h0 hm
    /-
      case refine_2.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne 0 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      h0 : Eq (HSub.hSub (m 0) (HMul.hMul 0 r)) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    rw [zero_mul, sub_zero] at h0
    /-
      case refine_2.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      r : Real
      m : Fin (Nat.succ 0).succ → Real
      hm : Or (Ne (m 0) 0) (Ne 0 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) y)) 0
      h0 : Eq (m 0) 0
      h1 : Eq (m 1) 0
      ⊢ False
    -/
    simp [h0] at hm
    /-
      🎉 no goals
    -/


/-- Adding a multiple of the first vector passed to `oangle` to the second vector does not change
the sign of the angle. -/
@[simp]
theorem oangle_sign_smul_add_right (x y : V) (r : ℝ) :
    (o.oangle x (r • x + y)).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul r x) y)).sign (o.oangle x y).sign
  -/
  by_cases h : o.oangle x y = 0 ∨ o.oangle x y = π
  · rwa [Real.Angle.sign_eq_zero_iff.2 h, Real.Angle.sign_eq_zero_iff,
      oangle_smul_add_right_eq_zero_or_eq_pi_iff]
  have h' : ∀ r' : ℝ, o.oangle x (r' • x + y) ≠ 0 ∧ o.oangle x (r' • x + y) ≠ π := by
    intro r'
    rwa [← o.oangle_smul_add_right_eq_zero_or_eq_pi_iff r', not_or] at h
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    h : Not (Or (Eq (o.oangle x y) 0) (Eq (o.oangle x y) ↑Real.pi))
    h' : ∀ (r' : Real), And (Ne (o.oangle x (HAdd.hAdd (HSMul.hSMul r' x) y)) 0) ( …
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul r x) y)).sign (o.oangle x y).sign
  -/
  let s : Set (V × V) := (fun r' : ℝ => (x, r' • x + y)) '' Set.univ
  have hc : IsConnected s := isConnected_univ.image _ (continuous_const.prod_mk
    ((continuous_id.smul continuous_const).add continuous_const)).continuousOn
  have hf : ContinuousOn (fun z : V × V => o.oangle z.1 z.2) s := by
    refine continuousOn_of_forall_continuousAt fun z hz => o.continuousAt_oangle ?_ ?_
    all_goals
      simp_rw [s, Set.mem_image] at hz
      obtain ⟨r', -, rfl⟩ := hz
      simp only [Prod.fst, Prod.snd]
      intro hz
    · simpa [hz] using (h' 0).1
    · simpa [hz] using (h' r').1
  have hs : ∀ z : V × V, z ∈ s → o.oangle z.1 z.2 ≠ 0 ∧ o.oangle z.1 z.2 ≠ π := by
    intro z hz
    simp_rw [s, Set.mem_image] at hz
    obtain ⟨r', -, rfl⟩ := hz
    exact h' r'
  have hx : (x, y) ∈ s := by
    convert Set.mem_image_of_mem (fun r' : ℝ => (x, r' • x + y)) (Set.mem_univ 0)
    simp
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    h : Not (Or (Eq (o.oangle x y) 0) (Eq (o.oangle x y) ↑Real.pi))
    h' : ∀ (r' : Real), And (Ne (o.oangle x (HAdd.hAdd (HSMul.hSMul r' x) y)) 0) ( …
    s : Set (Prod V V) := Set.image (fun r' => { fst := x, snd := HAdd.hAdd (HSMul …
    hc : IsConnected s
    hf : ContinuousOn (fun z => o.oangle z.1 z.2) s
    hs : ∀ (z : Prod V V), Membership.mem s z → And (Ne (o.oangle z.1 z.2) 0) (Ne  …
    hx : Membership.mem s { fst := x, snd := y }
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul r x) y)).sign (o.oangle x y).sign
  -/
  have hy : (x, r • x + y) ∈ s := Set.mem_image_of_mem _ (Set.mem_univ _)
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    h : Not (Or (Eq (o.oangle x y) 0) (Eq (o.oangle x y) ↑Real.pi))
    h' : ∀ (r' : Real), And (Ne (o.oangle x (HAdd.hAdd (HSMul.hSMul r' x) y)) 0) ( …
    s : Set (Prod V V) := Set.image (fun r' => { fst := x, snd := HAdd.hAdd (HSMul …
    hc : IsConnected s
    hf : ContinuousOn (fun z => o.oangle z.1 z.2) s
    hs : ∀ (z : Prod V V), Membership.mem s z → And (Ne (o.oangle z.1 z.2) 0) (Ne  …
    hx : Membership.mem s { fst := x, snd := y }
    hy : Membership.mem s { fst := x, snd := HAdd.hAdd (HSMul.hSMul r x) y }
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul r x) y)).sign (o.oangle x y).sign
  -/
  convert Real.Angle.sign_eq_of_continuousOn hc hf hs hx hy
  /-
    🎉 no goals
  -/


/-- Adding a multiple of the second vector passed to `oangle` to the first vector does not change
the sign of the angle. -/
@[simp]
theorem oangle_sign_add_smul_left (x y : V) (r : ℝ) :
    (o.oangle (x + r • y) y).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle (HAdd.hAdd x (HSMul.hSMul r y)) y).sign (o.oangle x y).sign
  -/
  simp_rw [o.oangle_rev y, Real.Angle.sign_neg, add_comm x, oangle_sign_smul_add_right]
  /-
    🎉 no goals
  -/


/-- Subtracting a multiple of the first vector passed to `oangle` from the second vector does
not change the sign of the angle. -/
@[simp]
theorem oangle_sign_sub_smul_right (x y : V) (r : ℝ) :
    (o.oangle x (y - r • x)).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle x (HSub.hSub y (HSMul.hSMul r x))).sign (o.oangle x y).sign
  -/
  rw [sub_eq_add_neg, ← neg_smul, add_comm, oangle_sign_smul_add_right]
  /-
    🎉 no goals
  -/


/-- Subtracting a multiple of the second vector passed to `oangle` from the first vector does
not change the sign of the angle. -/
@[simp]
theorem oangle_sign_sub_smul_left (x y : V) (r : ℝ) :
    (o.oangle (x - r • y) y).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle (HSub.hSub x (HSMul.hSMul r y)) y).sign (o.oangle x y).sign
  -/
  rw [sub_eq_add_neg, ← neg_smul, oangle_sign_add_smul_left]
  /-
    🎉 no goals
  -/


/-- Adding the first vector passed to `oangle` to the second vector does not change the sign of
the angle. -/
@[simp]
theorem oangle_sign_add_right (x y : V) : (o.oangle x (x + y)).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle x (HAdd.hAdd x y)).sign (o.oangle x y).sign
  -/
  rw [← o.oangle_sign_smul_add_right x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Adding the second vector passed to `oangle` to the first vector does not change the sign of
the angle. -/
@[simp]
theorem oangle_sign_add_left (x y : V) : (o.oangle (x + y) y).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (HAdd.hAdd x y) y).sign (o.oangle x y).sign
  -/
  rw [← o.oangle_sign_add_smul_left x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting the first vector passed to `oangle` from the second vector does not change the
sign of the angle. -/
@[simp]
theorem oangle_sign_sub_right (x y : V) : (o.oangle x (y - x)).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle x (HSub.hSub y x)).sign (o.oangle x y).sign
  -/
  rw [← o.oangle_sign_sub_smul_right x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting the second vector passed to `oangle` from the first vector does not change the
sign of the angle. -/
@[simp]
theorem oangle_sign_sub_left (x y : V) : (o.oangle (x - y) y).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (HSub.hSub x y) y).sign (o.oangle x y).sign
  -/
  rw [← o.oangle_sign_sub_smul_left x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting the second vector passed to `oangle` from a multiple of the first vector negates
the sign of the angle. -/
@[simp]
theorem oangle_sign_smul_sub_right (x y : V) (r : ℝ) :
    (o.oangle x (r • x - y)).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle x (HSub.hSub (HSMul.hSMul r x) y)).sign (Neg.neg (o.oangle x y) …
  -/
  rw [← oangle_sign_neg_right, sub_eq_add_neg, oangle_sign_smul_add_right]
  /-
    🎉 no goals
  -/


/-- Subtracting the first vector passed to `oangle` from a multiple of the second vector negates
the sign of the angle. -/
@[simp]
theorem oangle_sign_smul_sub_left (x y : V) (r : ℝ) :
    (o.oangle (r • y - x) y).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r : Real
    ⊢ Eq (o.oangle (HSub.hSub (HSMul.hSMul r y) x) y).sign (Neg.neg (o.oangle x y) …
  -/
  rw [← oangle_sign_neg_left, sub_eq_neg_add, oangle_sign_add_smul_left]
  /-
    🎉 no goals
  -/


/-- Subtracting the second vector passed to `oangle` from the first vector negates the sign of
the angle. -/
theorem oangle_sign_sub_right_eq_neg (x y : V) :
    (o.oangle x (x - y)).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle x (HSub.hSub x y)).sign (Neg.neg (o.oangle x y).sign)
  -/
  rw [← o.oangle_sign_smul_sub_right x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting the first vector passed to `oangle` from the second vector negates the sign of
the angle. -/
theorem oangle_sign_sub_left_eq_neg (x y : V) :
    (o.oangle (y - x) y).sign = -(o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (HSub.hSub y x) y).sign (Neg.neg (o.oangle x y).sign)
  -/
  rw [← o.oangle_sign_smul_sub_left x y 1, one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting the first vector passed to `oangle` from the second vector then swapping the
vectors does not change the sign of the angle. -/
@[simp]
theorem oangle_sign_sub_right_swap (x y : V) : (o.oangle y (y - x)).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle y (HSub.hSub y x)).sign (o.oangle x y).sign
  -/
  rw [oangle_sign_sub_right_eq_neg, o.oangle_rev y x, Real.Angle.sign_neg]
  /-
    🎉 no goals
  -/


/-- Subtracting the second vector passed to `oangle` from the first vector then swapping the
vectors does not change the sign of the angle. -/
@[simp]
theorem oangle_sign_sub_left_swap (x y : V) : (o.oangle (x - y) x).sign = (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle (HSub.hSub x y) x).sign (o.oangle x y).sign
  -/
  rw [oangle_sign_sub_left_eq_neg, o.oangle_rev y x, Real.Angle.sign_neg]
  /-
    🎉 no goals
  -/


/-- The sign of the angle between a vector, and a linear combination of that vector with a second
vector, is the sign of the factor by which the second vector is multiplied in that combination
multiplied by the sign of the angle between the two vectors. -/
theorem oangle_sign_smul_add_smul_right (x y : V) (r₁ r₂ : ℝ) :
    (o.oangle x (r₁ • x + r₂ • y)).sign = SignType.sign r₂ * (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r₁ r₂ : Real
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y))).sign (HMul …
  -/
  rw [← o.oangle_sign_smul_add_right x (r₁ • x + r₂ • y) (-r₁)]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r₁ r₂ : Real
    ⊢ Eq (o.oangle x (HAdd.hAdd (HSMul.hSMul (Neg.neg r₁) x) (HAdd.hAdd (HSMul.hSM …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The sign of the angle between a linear combination of two vectors and the second vector is
the sign of the factor by which the first vector is multiplied in that combination multiplied by
the sign of the angle between the two vectors. -/
theorem oangle_sign_smul_add_smul_left (x y : V) (r₁ r₂ : ℝ) :
    (o.oangle (r₁ • x + r₂ • y) y).sign = SignType.sign r₁ * (o.oangle x y).sign := by
  simp_rw [o.oangle_rev y, Real.Angle.sign_neg, add_comm (r₁ • x), oangle_sign_smul_add_smul_right,
    mul_neg]


/-- The sign of the angle between two linear combinations of two vectors is the sign of the
determinant of the factors in those combinations multiplied by the sign of the angle between the
two vectors. -/
theorem oangle_sign_smul_add_smul_smul_add_smul (x y : V) (r₁ r₂ r₃ r₄ : ℝ) :
    (o.oangle (r₁ • x + r₂ • y) (r₃ • x + r₄ • y)).sign =
      SignType.sign (r₁ * r₄ - r₂ * r₃) * (o.oangle x y).sign := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    r₁ r₂ r₃ r₄ : Real
    ⊢ Eq (o.oangle (HAdd.hAdd (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)) (HAdd.hAdd (H …
  -/
  by_cases hr₁ : r₁ = 0
  · rw [hr₁, zero_smul, zero_mul, zero_add, zero_sub, Left.sign_neg,
      oangle_sign_smul_left, add_comm, oangle_sign_smul_add_smul_right, oangle_rev,
      Real.Angle.sign_neg, sign_mul, mul_neg, mul_neg, neg_mul, mul_assoc]
  · rw [← o.oangle_sign_smul_add_right (r₁ • x + r₂ • y) (r₃ • x + r₄ • y) (-r₃ / r₁), smul_add,
      smul_smul, smul_smul, div_mul_cancel₀ _ hr₁, neg_smul, ← add_assoc, add_comm (-(r₃ • x)), ←
      sub_eq_add_neg, sub_add_cancel, ← add_smul, oangle_sign_smul_right,
      oangle_sign_smul_add_smul_left, ← mul_assoc, ← sign_mul, add_mul, mul_assoc, mul_comm r₂ r₁, ←
      mul_assoc, div_mul_cancel₀ _ hr₁, add_comm, neg_mul, ← sub_eq_add_neg, mul_comm r₄,
      mul_comm r₃]


/-- A base angle of an isosceles triangle is acute, oriented vector angle form. -/
theorem abs_oangle_sub_left_toReal_lt_pi_div_two {x y : V} (h : ‖x‖ = ‖y‖) :
    |(o.oangle (y - x) y).toReal| < π / 2 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
  -/
  by_cases hn : x = y; · simp [hn, div_pos, Real.pi_pos]
                         /-
                           🎉 no goals
                         -/
  have hs : ((2 : ℤ) • o.oangle (y - x) y).sign = (o.oangle (y - x) y).sign := by
    conv_rhs => rw [oangle_sign_sub_left_swap]
    rw [o.oangle_eq_pi_sub_two_zsmul_oangle_sub_of_norm_eq hn h, Real.Angle.sign_pi_sub]
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (Norm.norm x) (Norm.norm y)
    hn : Not (Eq x y)
    hs : Eq (HSMul.hSMul 2 (o.oangle (HSub.hSub y x) y)).sign (o.oangle (HSub.hSub …
    ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
  -/
  rw [Real.Angle.sign_two_zsmul_eq_sign_iff] at hs
  /-
    case neg
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    h : Eq (Norm.norm x) (Norm.norm y)
    hn : Not (Eq x y)
    hs : Or (Eq (o.oangle (HSub.hSub y x) y) ↑Real.pi) (LT.lt (abs (o.oangle (HSub …
    ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
  -/
  rcases hs with (hs | hs)
    /-
      case neg.inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hs : Eq (o.oangle (HSub.hSub y x) y) ↑Real.pi
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
  · rw [oangle_eq_pi_iff_oangle_rev_eq_pi, oangle_eq_pi_iff_sameRay_neg, neg_sub] at hs
    /-
      case neg.inl
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hs : And (Ne y 0) (And (Ne (HSub.hSub y x) 0) (SameRay Real y (HSub.hSub x y)))
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rcases hs with ⟨hy, -, hr⟩
    /-
      case neg.inl.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hy : Ne y 0
      hr : SameRay Real y (HSub.hSub x y)
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rw [← exists_nonneg_left_iff_sameRay hy] at hr
    /-
      case neg.inl.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hy : Ne y 0
      hr : Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r y) (HSub.hSub x y))
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rcases hr with ⟨r, hr0, hr⟩
    /-
      case neg.inl.intro.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hy : Ne y 0
      r : Real
      hr0 : LE.le 0 r
      hr : Eq (HSMul.hSMul r y) (HSub.hSub x y)
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rw [eq_sub_iff_add_eq] at hr
    /-
      case neg.inl.intro.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hy : Ne y 0
      r : Real
      hr0 : LE.le 0 r
      hr : Eq (HAdd.hAdd (HSMul.hSMul r y) y) x
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    nth_rw 2 [← one_smul ℝ y] at hr
    /-
      case neg.inl.intro.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hy : Ne y 0
      r : Real
      hr0 : LE.le 0 r
      hr : Eq (HAdd.hAdd (HSMul.hSMul r y) (HSMul.hSMul 1 y)) x
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rw [← add_smul] at hr
    rw [← hr, norm_smul, Real.norm_eq_abs, abs_of_pos (Left.add_pos_of_nonneg_of_pos hr0 one_pos),
      mul_left_eq_self₀, or_iff_left (norm_ne_zero_iff.2 hy), add_left_eq_self] at h
    /-
      case neg.inl.intro.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hn : Not (Eq x y)
      hy : Ne y 0
      r : Real
      h : Eq r 0
      hr0 : LE.le 0 r
      hr : Eq (HSMul.hSMul (HAdd.hAdd r 1) y) x
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    rw [h, zero_add, one_smul] at hr
    /-
      case neg.inl.intro.intro.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hn : Not (Eq x y)
      hy : Ne y 0
      r : Real
      h : Eq r 0
      hr0 : LE.le 0 r
      hr : Eq y x
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
    exact False.elim (hn hr.symm)
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      hn : Not (Eq x y)
      hs : LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
      ⊢ LT.lt (abs (o.oangle (HSub.hSub y x) y).toReal) (HDiv.hDiv Real.pi 2)
    -/
  · exact hs
    /-
      🎉 no goals
    -/


/-- A base angle of an isosceles triangle is acute, oriented vector angle form. -/
theorem abs_oangle_sub_right_toReal_lt_pi_div_two {x y : V} (h : ‖x‖ = ‖y‖) :
    |(o.oangle x (x - y)).toReal| < π / 2 :=
  (o.oangle_sub_eq_oangle_sub_rev_of_norm_eq h).symm ▸ o.abs_oangle_sub_left_toReal_lt_pi_div_two h


