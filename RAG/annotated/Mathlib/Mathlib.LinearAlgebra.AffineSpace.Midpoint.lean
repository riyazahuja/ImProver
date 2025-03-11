/-- `midpoint x y` is the midpoint of the segment `[x, y]`. -/
def midpoint (x y : P) : P :=
  lineMap x y (⅟ 2 : R)


@[simp]
theorem AffineMap.map_midpoint (f : P →ᵃ[R] P') (a b : P) :
    f (midpoint R a b) = midpoint R (f a) (f b) :=
  f.apply_lineMap a b _


@[simp]
theorem AffineEquiv.map_midpoint (f : P ≃ᵃ[R] P') (a b : P) :
    f (midpoint R a b) = midpoint R (f a) (f b) :=
  f.apply_lineMap a b _


theorem AffineEquiv.pointReflection_midpoint_left (x y : P) :
    pointReflection R (midpoint R x y) x = y := by
  rw [midpoint, pointReflection_apply, lineMap_apply, vadd_vsub, vadd_vadd, ← add_smul, ← two_mul,
    mul_invOf_self, one_smul, vsub_vadd]


@[simp] -- Porting note: added variant with `Equiv.pointReflection` for `simp`
theorem Equiv.pointReflection_midpoint_left (x y : P) :
    (Equiv.pointReflection (midpoint R x y)) x = y := by
  rw [midpoint, pointReflection_apply, lineMap_apply, vadd_vsub, vadd_vadd, ← add_smul, ← two_mul,
    mul_invOf_self, one_smul, vsub_vadd]


theorem midpoint_comm (x y : P) : midpoint R x y = midpoint R y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq (midpoint R x y) (midpoint R y x)
  -/
  rw [midpoint, ← lineMap_apply_one_sub, one_sub_invOf_two, midpoint]
  /-
    🎉 no goals
  -/


theorem AffineEquiv.pointReflection_midpoint_right (x y : P) :
    pointReflection R (midpoint R x y) y = x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq ((AffineEquiv.pointReflection R (midpoint R x y)) y) x
  -/
  rw [midpoint_comm, AffineEquiv.pointReflection_midpoint_left]
  /-
    🎉 no goals
  -/


@[simp] -- Porting note: added variant with `Equiv.pointReflection` for `simp`
theorem Equiv.pointReflection_midpoint_right (x y : P) :
    (Equiv.pointReflection (midpoint R x y)) y = x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq ((Equiv.pointReflection (midpoint R x y)) y) x
  -/
  rw [midpoint_comm, Equiv.pointReflection_midpoint_left]
  /-
    🎉 no goals
  -/


theorem midpoint_vsub_midpoint (p₁ p₂ p₃ p₄ : P) :
    midpoint R p₁ p₂ -ᵥ midpoint R p₃ p₄ = midpoint R (p₁ -ᵥ p₃) (p₂ -ᵥ p₄) :=
  lineMap_vsub_lineMap _ _ _ _ _


theorem midpoint_vadd_midpoint (v v' : V) (p p' : P) :
    midpoint R v v' +ᵥ midpoint R p p' = midpoint R (v +ᵥ p) (v' +ᵥ p') :=
  lineMap_vadd_lineMap _ _ _ _ _


theorem midpoint_eq_iff {x y z : P} : midpoint R x y = z ↔ pointReflection R z x = y :=
  eq_comm.trans
    ((injective_pointReflection_left_of_module R x).eq_iff'
        (AffineEquiv.pointReflection_midpoint_left x y)).symm


@[simp]
theorem midpoint_pointReflection_left (x y : P) :
    midpoint R (Equiv.pointReflection x y) y = x :=
  midpoint_eq_iff.2 <| Equiv.pointReflection_involutive _ _


@[simp]
theorem midpoint_pointReflection_right (x y : P) :
    midpoint R y (Equiv.pointReflection x y) = x :=
  midpoint_eq_iff.2 rfl


@[simp]
theorem midpoint_vsub_left (p₁ p₂ : P) : midpoint R p₁ p₂ -ᵥ p₁ = (⅟ 2 : R) • (p₂ -ᵥ p₁) :=
  lineMap_vsub_left _ _ _


@[simp]
theorem midpoint_vsub_right (p₁ p₂ : P) : midpoint R p₁ p₂ -ᵥ p₂ = (⅟ 2 : R) • (p₁ -ᵥ p₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ Eq (VSub.vsub (midpoint R p₁ p₂) p₂) (HSMul.hSMul (Invertible.invOf 2) (VSub …
  -/
  rw [midpoint_comm, midpoint_vsub_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem left_vsub_midpoint (p₁ p₂ : P) : p₁ -ᵥ midpoint R p₁ p₂ = (⅟ 2 : R) • (p₁ -ᵥ p₂) :=
  left_vsub_lineMap _ _ _


@[simp]
theorem right_vsub_midpoint (p₁ p₂ : P) : p₂ -ᵥ midpoint R p₁ p₂ = (⅟ 2 : R) • (p₂ -ᵥ p₁) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ Eq (VSub.vsub p₂ (midpoint R p₁ p₂)) (HSMul.hSMul (Invertible.invOf 2) (VSub …
  -/
  rw [midpoint_comm, left_vsub_midpoint]
  /-
    🎉 no goals
  -/


theorem midpoint_vsub (p₁ p₂ p : P) :
    midpoint R p₁ p₂ -ᵥ p = (⅟ 2 : R) • (p₁ -ᵥ p) + (⅟ 2 : R) • (p₂ -ᵥ p) := by
  rw [← vsub_sub_vsub_cancel_right p₁ p p₂, smul_sub, sub_eq_add_neg, ← smul_neg,
    neg_vsub_eq_vsub_rev, add_assoc, invOf_two_smul_add_invOf_two_smul, ← vadd_vsub_assoc,
    midpoint_comm, midpoint, lineMap_apply]


theorem vsub_midpoint (p₁ p₂ p : P) :
    p -ᵥ midpoint R p₁ p₂ = (⅟ 2 : R) • (p -ᵥ p₁) + (⅟ 2 : R) • (p -ᵥ p₂) := by
  rw [← neg_vsub_eq_vsub_rev, midpoint_vsub, neg_add, ← smul_neg, ← smul_neg, neg_vsub_eq_vsub_rev,
    neg_vsub_eq_vsub_rev]


@[simp]
theorem midpoint_sub_left (v₁ v₂ : V) : midpoint R v₁ v₂ - v₁ = (⅟ 2 : R) • (v₂ - v₁) :=
  midpoint_vsub_left v₁ v₂


@[simp]
theorem midpoint_sub_right (v₁ v₂ : V) : midpoint R v₁ v₂ - v₂ = (⅟ 2 : R) • (v₁ - v₂) :=
  midpoint_vsub_right v₁ v₂


@[simp]
theorem left_sub_midpoint (v₁ v₂ : V) : v₁ - midpoint R v₁ v₂ = (⅟ 2 : R) • (v₁ - v₂) :=
  left_vsub_midpoint v₁ v₂


@[simp]
theorem right_sub_midpoint (v₁ v₂ : V) : v₂ - midpoint R v₁ v₂ = (⅟ 2 : R) • (v₂ - v₁) :=
  right_vsub_midpoint v₁ v₂


@[simp]
theorem midpoint_eq_left_iff {x y : P} : midpoint R x y = x ↔ x = y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Iff (Eq (midpoint R x y) x) (Eq x y)
  -/
  rw [midpoint_eq_iff, pointReflection_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem left_eq_midpoint_iff {x y : P} : x = midpoint R x y ↔ x = y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Iff (Eq x (midpoint R x y)) (Eq x y)
  -/
  rw [eq_comm, midpoint_eq_left_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem midpoint_eq_right_iff {x y : P} : midpoint R x y = y ↔ x = y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Iff (Eq (midpoint R x y) y) (Eq x y)
  -/
  rw [midpoint_comm, midpoint_eq_left_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem right_eq_midpoint_iff {x y : P} : y = midpoint R x y ↔ x = y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : Ring R
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Iff (Eq y (midpoint R x y)) (Eq x y)
  -/
  rw [eq_comm, midpoint_eq_right_iff]
  /-
    🎉 no goals
  -/


theorem midpoint_eq_midpoint_iff_vsub_eq_vsub {x x' y y' : P} :
    midpoint R x y = midpoint R x' y' ↔ x -ᵥ x' = y' -ᵥ y := by
  rw [← @vsub_eq_zero_iff_eq V, midpoint_vsub_midpoint, midpoint_eq_iff, pointReflection_apply,
    vsub_eq_sub, zero_sub, vadd_eq_add, add_zero, neg_eq_iff_eq_neg, neg_vsub_eq_vsub_rev]


theorem midpoint_eq_iff' {x y z : P} : midpoint R x y = z ↔ Equiv.pointReflection z x = y :=
  midpoint_eq_iff


/-- `midpoint` does not depend on the ring `R`. -/
theorem midpoint_unique (R' : Type*) [Ring R'] [Invertible (2 : R')] [Module R' V] (x y : P) :
    midpoint R x y = midpoint R' x y :=
  (midpoint_eq_iff' R).2 <| (midpoint_eq_iff' R').1 rfl


@[simp]
theorem midpoint_self (x : P) : midpoint R x x = x :=
  lineMap_same_apply _ _


@[simp]
theorem midpoint_add_self (x y : V) : midpoint R x y + midpoint R x y = x + y :=
  calc
                                                                              /-
                                                                                R : Type u_1
                                                                                V : Type u_2
                                                                                inst✝³ : Ring R
                                                                                inst✝² : Invertible 2
                                                                                inst✝¹ : AddCommGroup V
                                                                                inst✝ : Module R V
                                                                                x y : V
                                                                                ⊢ Eq (HVAdd.hVAdd (midpoint R x y) (midpoint R x y)) (HVAdd.hVAdd (midpoint R  …
                                                                              -/
    midpoint R x y +ᵥ midpoint R x y = midpoint R x y +ᵥ midpoint R y x := by rw [midpoint_comm]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                    /-
                      R : Type u_1
                      V : Type u_2
                      inst✝³ : Ring R
                      inst✝² : Invertible 2
                      inst✝¹ : AddCommGroup V
                      inst✝ : Module R V
                      x y : V
                      ⊢ Eq (HVAdd.hVAdd (midpoint R x y) (midpoint R y x)) (HAdd.hAdd x y)
                    -/
    _ = x + y := by rw [midpoint_vadd_midpoint, vadd_eq_add, vadd_eq_add, add_comm, midpoint_self]
                    /-
                      🎉 no goals
                    -/


theorem midpoint_zero_add (x y : V) : midpoint R 0 (x + y) = midpoint R x y :=
                                                    /-
                                                      R : Type u_1
                                                      V : Type u_2
                                                      inst✝³ : Ring R
                                                      inst✝² : Invertible 2
                                                      inst✝¹ : AddCommGroup V
                                                      inst✝ : Module R V
                                                      x y : V
                                                      ⊢ Eq (VSub.vsub 0 x) (VSub.vsub y (HAdd.hAdd x y))
                                                    -/
  (midpoint_eq_midpoint_iff_vsub_eq_vsub R).2 <| by simp [sub_add_eq_sub_sub_swap]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem midpoint_eq_smul_add (x y : V) : midpoint R x y = (⅟ 2 : R) • (x + y) := by
  rw [midpoint_eq_iff, pointReflection_apply, vsub_eq_sub, vadd_eq_add, sub_add_eq_add_sub, ←
    two_smul R, smul_smul, mul_invOf_self, one_smul, add_sub_cancel_left]


@[simp]
theorem midpoint_self_neg (x : V) : midpoint R x (-x) = 0 := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝³ : Ring R
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    x : V
    ⊢ Eq (midpoint R x (Neg.neg x)) 0
  -/
  rw [midpoint_eq_smul_add, add_neg_cancel, smul_zero]
  /-
    🎉 no goals
  -/


@[simp]
                                                                /-
                                                                  R : Type u_1
                                                                  V : Type u_2
                                                                  inst✝³ : Ring R
                                                                  inst✝² : Invertible 2
                                                                  inst✝¹ : AddCommGroup V
                                                                  inst✝ : Module R V
                                                                  x : V
                                                                  ⊢ Eq (midpoint R (Neg.neg x) x) 0
                                                                -/
theorem midpoint_neg_self (x : V) : midpoint R (-x) x = 0 := by simpa using midpoint_self_neg R (-x)
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem midpoint_sub_add (x y : V) : midpoint R (x - y) (x + y) = x := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝³ : Ring R
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    x y : V
    ⊢ Eq (midpoint R (HSub.hSub x y) (HAdd.hAdd x y)) x
  -/
  rw [sub_eq_add_neg, ← vadd_eq_add, ← vadd_eq_add, ← midpoint_vadd_midpoint]; simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem midpoint_add_sub (x y : V) : midpoint R (x + y) (x - y) = x := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝³ : Ring R
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    x y : V
    ⊢ Eq (midpoint R (HAdd.hAdd x y) (HSub.hSub x y)) x
  -/
  rw [midpoint_comm]; simp
                      /-
                        🎉 no goals
                      -/


/-- A map `f : E → F` sending zero to zero and midpoints to midpoints is an `AddMonoidHom`. -/
def ofMapMidpoint (f : E → F) (h0 : f 0 = 0)
    (hm : ∀ x y, f (midpoint R x y) = midpoint R' (f x) (f y)) : E →+ F where
  toFun := f
  map_zero' := h0
  map_add' x y :=
    calc
                                        /-
                                          R : Type u_1
                                          R' : Type u_2
                                          E : Type u_3
                                          F : Type u_4
                                          inst✝⁷ : Ring R
                                          inst✝⁶ : Invertible 2
                                          inst✝⁵ : AddCommGroup E
                                          inst✝⁴ : Module R E
                                          inst✝³ : Ring R'
                                          inst✝² : Invertible 2
                                          inst✝¹ : AddCommGroup F
                                          inst✝ : Module R' F
                                          f : E → F
                                          h0 : Eq (f 0) 0
                                          hm : ∀ (x y : E), Eq (f (midpoint R x y)) (midpoint R' (f x) (f y))
                                          x y : E
                                          ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f 0) (f (HAdd.hAdd x y)))
                                        -/
      f (x + y) = f 0 + f (x + y) := by rw [h0, zero_add]
                                        /-
                                          🎉 no goals
                                        -/
      _ = midpoint R' (f 0) (f (x + y)) + midpoint R' (f 0) (f (x + y)) :=
        (midpoint_add_self _ _ _).symm
                                                        /-
                                                          R : Type u_1
                                                          R' : Type u_2
                                                          E : Type u_3
                                                          F : Type u_4
                                                          inst✝⁷ : Ring R
                                                          inst✝⁶ : Invertible 2
                                                          inst✝⁵ : AddCommGroup E
                                                          inst✝⁴ : Module R E
                                                          inst✝³ : Ring R'
                                                          inst✝² : Invertible 2
                                                          inst✝¹ : AddCommGroup F
                                                          inst✝ : Module R' F
                                                          f : E → F
                                                          h0 : Eq (f 0) 0
                                                          hm : ∀ (x y : E), Eq (f (midpoint R x y)) (midpoint R' (f x) (f y))
                                                          x y : E
                                                          ⊢ Eq (HAdd.hAdd (midpoint R' (f 0) (f (HAdd.hAdd x y))) (midpoint R' (f 0) (f  …
                                                        -/
      _ = f (midpoint R x y) + f (midpoint R x y) := by rw [← hm, midpoint_zero_add]
                                                        /-
                                                          🎉 no goals
                                                        -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            inst✝⁷ : Ring R
                            inst✝⁶ : Invertible 2
                            inst✝⁵ : AddCommGroup E
                            inst✝⁴ : Module R E
                            inst✝³ : Ring R'
                            inst✝² : Invertible 2
                            inst✝¹ : AddCommGroup F
                            inst✝ : Module R' F
                            f : E → F
                            h0 : Eq (f 0) 0
                            hm : ∀ (x y : E), Eq (f (midpoint R x y)) (midpoint R' (f x) (f y))
                            x y : E
                            ⊢ Eq (HAdd.hAdd (f (midpoint R x y)) (f (midpoint R x y))) (HAdd.hAdd (f x) (f …
                          -/
      _ = f x + f y := by rw [hm, midpoint_add_self]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem coe_ofMapMidpoint (f : E → F) (h0 : f 0 = 0)
    (hm : ∀ x y, f (midpoint R x y) = midpoint R' (f x) (f y)) :
    ⇑(ofMapMidpoint R R' f h0 hm) = f :=
  rfl


