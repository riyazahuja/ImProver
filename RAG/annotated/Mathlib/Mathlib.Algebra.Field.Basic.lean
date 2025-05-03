                                                                /-
                                                                  K : Type u_1
                                                                  inst✝ : DivisionSemiring K
                                                                  a b c : K
                                                                  ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) c) (HAdd.hAdd (HDiv.hDiv a c) (HDiv.hDiv b c))
                                                                -/
theorem add_div (a b c : K) : (a + b) / c = a / c + b / c := by simp_rw [div_eq_mul_inv, add_mul]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[field_simps]
theorem div_add_div_same (a b c : K) : a / c + b / c = (a + b) / c :=
  (add_div _ _ _).symm


                                                                 /-
                                                                   K : Type u_1
                                                                   inst✝ : DivisionSemiring K
                                                                   a b : K
                                                                   h : Ne b 0
                                                                   ⊢ Eq (HDiv.hDiv (HAdd.hAdd b a) b) (HAdd.hAdd 1 (HDiv.hDiv a b))
                                                                 -/
theorem same_add_div (h : b ≠ 0) : (b + a) / b = 1 + a / b := by rw [← div_self h, add_div]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                 /-
                                                                   K : Type u_1
                                                                   inst✝ : DivisionSemiring K
                                                                   a b : K
                                                                   h : Ne b 0
                                                                   ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) b) (HAdd.hAdd (HDiv.hDiv a b) 1)
                                                                 -/
theorem div_add_same (h : b ≠ 0) : (a + b) / b = a / b + 1 := by rw [← div_self h, add_div]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem one_add_div (h : b ≠ 0) : 1 + a / b = (b + a) / b :=
  (same_add_div h).symm


theorem div_add_one (h : b ≠ 0) : a / b + 1 = (a + b) / b :=
  (div_add_same h).symm


/-- See `inv_add_inv` for the more convenient version when `K` is commutative. -/
theorem inv_add_inv' (ha : a ≠ 0) (hb : b ≠ 0) :
    a⁻¹ + b⁻¹ = a⁻¹ * (a + b) * b⁻¹ :=
  let _ := invertibleOfNonzero ha; let _ := invertibleOfNonzero hb; invOf_add_invOf a b


theorem one_div_mul_add_mul_one_div_eq_one_div_add_one_div (ha : a ≠ 0) (hb : b ≠ 0) :
    1 / a * (a + b) * (1 / b) = 1 / a + 1 / b := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b : K
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv 1 a) (HAdd.hAdd a b)) (HDiv.hDiv 1 b)) ( …
  -/
  simpa only [one_div] using (inv_add_inv' ha hb).symm
  /-
    🎉 no goals
  -/


theorem add_div_eq_mul_add_div (a b : K) (hc : c ≠ 0) : a + b / c = (a * c + b) / c :=
                                 /-
                                   K : Type u_1
                                   inst✝ : DivisionSemiring K
                                   c a b : K
                                   hc : Ne c 0
                                   ⊢ Eq (HMul.hMul (HAdd.hAdd a (HDiv.hDiv b c)) c) (HAdd.hAdd (HMul.hMul a c) b)
                                 -/
  (eq_div_iff_mul_eq hc).2 <| by rw [right_distrib, div_mul_cancel₀ _ hc]
                                 /-
                                   🎉 no goals
                                 -/


@[field_simps]
theorem add_div' (a b c : K) (hc : c ≠ 0) : b + a / c = (b * c + a) / c := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b c : K
    hc : Ne c 0
    ⊢ Eq (HAdd.hAdd b (HDiv.hDiv a c)) (HDiv.hDiv (HAdd.hAdd (HMul.hMul b c) a) c)
  -/
  rw [add_div, mul_div_cancel_right₀ _ hc]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem div_add' (a b c : K) (hc : c ≠ 0) : a / c + b = (a + b * c) / c := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b c : K
    hc : Ne c 0
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv a c) b) (HDiv.hDiv (HAdd.hAdd a (HMul.hMul b c)) c)
  -/
  rwa [add_comm, add_div', add_comm]
  /-
    🎉 no goals
  -/


protected theorem Commute.div_add_div (hbc : Commute b c) (hbd : Commute b d) (hb : b ≠ 0)
    (hd : d ≠ 0) : a / b + c / d = (a * d + b * c) / (b * d) := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b c d : K
    hbc : Commute b c
    hbd : Commute b d
    hb : Ne b 0
    hd : Ne d 0
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv a b) (HDiv.hDiv c d)) (HDiv.hDiv (HAdd.hAdd (HMul.h …
  -/
  rw [add_div, mul_div_mul_right _ b hd, hbc.eq, hbd.eq, mul_div_mul_right c d hb]
  /-
    🎉 no goals
  -/


protected theorem Commute.one_div_add_one_div (hab : Commute a b) (ha : a ≠ 0) (hb : b ≠ 0) :
    1 / a + 1 / b = (a + b) / (a * b) := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b : K
    hab : Commute a b
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv 1 a) (HDiv.hDiv 1 b)) (HDiv.hDiv (HAdd.hAdd a b) (H …
  -/
  rw [(Commute.one_right a).div_add_div hab ha hb, one_mul, mul_one, add_comm]
  /-
    🎉 no goals
  -/


protected theorem Commute.inv_add_inv (hab : Commute a b) (ha : a ≠ 0) (hb : b ≠ 0) :
    a⁻¹ + b⁻¹ = (a + b) / (a * b) := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    a b : K
    hab : Commute a b
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HAdd.hAdd (Inv.inv a) (Inv.inv b)) (HDiv.hDiv (HAdd.hAdd a b) (HMul.hMul …
  -/
  rw [inv_eq_one_div, inv_eq_one_div, hab.one_div_add_one_div ha hb]
  /-
    🎉 no goals
  -/


theorem one_div_neg_one_eq_neg_one : (1 : K) / -1 = -1 :=
                                 /-
                                   K : Type u_1
                                   inst✝¹ : DivisionMonoid K
                                   inst✝ : HasDistribNeg K
                                   ⊢ Eq (HMul.hMul (-1) (-1)) 1
                                 -/
  have : -1 * -1 = (1 : K) := by rw [neg_mul_neg, one_mul]
                                 /-
                                   🎉 no goals
                                 -/
  Eq.symm (eq_one_div_of_mul_eq_one_right this)


theorem one_div_neg_eq_neg_one_div (a : K) : 1 / -a = -(1 / a) :=
  calc
                                /-
                                  K : Type u_1
                                  inst✝¹ : DivisionMonoid K
                                  inst✝ : HasDistribNeg K
                                  a : K
                                  ⊢ Eq (HDiv.hDiv 1 (Neg.neg a)) (HDiv.hDiv 1 (HMul.hMul (-1) a))
                                -/
    1 / -a = 1 / (-1 * a) := by rw [neg_eq_neg_one_mul]
                                /-
                                  🎉 no goals
                                -/
                               /-
                                 K : Type u_1
                                 inst✝¹ : DivisionMonoid K
                                 inst✝ : HasDistribNeg K
                                 a : K
                                 ⊢ Eq (HDiv.hDiv 1 (HMul.hMul (-1) a)) (HMul.hMul (HDiv.hDiv 1 a) (HDiv.hDiv 1  …
                               -/
    _ = 1 / a * (1 / -1) := by rw [one_div_mul_one_div_rev]
                               /-
                                 🎉 no goals
                               -/
                         /-
                           K : Type u_1
                           inst✝¹ : DivisionMonoid K
                           inst✝ : HasDistribNeg K
                           a : K
                           ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) (HDiv.hDiv 1 (-1))) (HMul.hMul (HDiv.hDiv 1 a) …
                         -/
    _ = 1 / a * -1 := by rw [one_div_neg_one_eq_neg_one]
                         /-
                           🎉 no goals
                         -/
                       /-
                         K : Type u_1
                         inst✝¹ : DivisionMonoid K
                         inst✝ : HasDistribNeg K
                         a : K
                         ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) (-1)) (Neg.neg (HDiv.hDiv 1 a))
                       -/
    _ = -(1 / a) := by rw [mul_neg, mul_one]
                       /-
                         🎉 no goals
                       -/


theorem div_neg_eq_neg_div (a b : K) : b / -a = -(b / a) :=
  calc
                                /-
                                  K : Type u_1
                                  inst✝¹ : DivisionMonoid K
                                  inst✝ : HasDistribNeg K
                                  a b : K
                                  ⊢ Eq (HDiv.hDiv b (Neg.neg a)) (HMul.hMul b (HDiv.hDiv 1 (Neg.neg a)))
                                -/
    b / -a = b * (1 / -a) := by rw [← inv_eq_one_div, division_def]
                                /-
                                  🎉 no goals
                                -/
                           /-
                             K : Type u_1
                             inst✝¹ : DivisionMonoid K
                             inst✝ : HasDistribNeg K
                             a b : K
                             ⊢ Eq (HMul.hMul b (HDiv.hDiv 1 (Neg.neg a))) (HMul.hMul b (Neg.neg (HDiv.hDiv  …
                           -/
    _ = b * -(1 / a) := by rw [one_div_neg_eq_neg_one_div]
                           /-
                             🎉 no goals
                           -/
                             /-
                               K : Type u_1
                               inst✝¹ : DivisionMonoid K
                               inst✝ : HasDistribNeg K
                               a b : K
                               ⊢ Eq (HMul.hMul b (Neg.neg (HDiv.hDiv 1 a))) (Neg.neg (HMul.hMul b (HDiv.hDiv  …
                             -/
    _ = -(b * (1 / a)) := by rw [neg_mul_eq_mul_neg]
                             /-
                               🎉 no goals
                             -/
                       /-
                         K : Type u_1
                         inst✝¹ : DivisionMonoid K
                         inst✝ : HasDistribNeg K
                         a b : K
                         ⊢ Eq (Neg.neg (HMul.hMul b (HDiv.hDiv 1 a))) (Neg.neg (HDiv.hDiv b a))
                       -/
    _ = -(b / a) := by rw [mul_one_div]
                       /-
                         🎉 no goals
                       -/


theorem neg_div (a b : K) : -b / a = -(b / a) := by
  /-
    K : Type u_1
    inst✝¹ : DivisionMonoid K
    inst✝ : HasDistribNeg K
    a b : K
    ⊢ Eq (HDiv.hDiv (Neg.neg b) a) (Neg.neg (HDiv.hDiv b a))
  -/
  rw [neg_eq_neg_one_mul, mul_div_assoc, ← neg_eq_neg_one_mul]
  /-
    🎉 no goals
  -/


@[field_simps]
                                                     /-
                                                       K : Type u_1
                                                       inst✝¹ : DivisionMonoid K
                                                       inst✝ : HasDistribNeg K
                                                       a b : K
                                                       ⊢ Eq (Neg.neg (HDiv.hDiv b a)) (HDiv.hDiv (Neg.neg b) a)
                                                     -/
theorem neg_div' (a b : K) : -(b / a) = -b / a := by simp [neg_div]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                                         /-
                                                           K : Type u_1
                                                           inst✝¹ : DivisionMonoid K
                                                           inst✝ : HasDistribNeg K
                                                           a b : K
                                                           ⊢ Eq (HDiv.hDiv (Neg.neg a) (Neg.neg b)) (HDiv.hDiv a b)
                                                         -/
theorem neg_div_neg_eq (a b : K) : -a / -b = a / b := by rw [div_neg_eq_neg_div, neg_div, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                      /-
                                        K : Type u_1
                                        inst✝¹ : DivisionMonoid K
                                        inst✝ : HasDistribNeg K
                                        a : K
                                        ⊢ Eq (Neg.neg (Inv.inv a)) (Inv.inv (Neg.neg a))
                                      -/
theorem neg_inv : -a⁻¹ = (-a)⁻¹ := by rw [inv_eq_one_div, inv_eq_one_div, div_neg_eq_neg_div]
                                      /-
                                        🎉 no goals
                                      -/


                                                  /-
                                                    K : Type u_1
                                                    inst✝¹ : DivisionMonoid K
                                                    inst✝ : HasDistribNeg K
                                                    b a : K
                                                    ⊢ Eq (HDiv.hDiv a (Neg.neg b)) (Neg.neg (HDiv.hDiv a b))
                                                  -/
theorem div_neg (a : K) : a / -b = -(a / b) := by rw [← div_neg_eq_neg_div]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                      /-
                                        K : Type u_1
                                        inst✝¹ : DivisionMonoid K
                                        inst✝ : HasDistribNeg K
                                        a : K
                                        ⊢ Eq (Inv.inv (Neg.neg a)) (Neg.neg (Inv.inv a))
                                      -/
theorem inv_neg : (-a)⁻¹ = -a⁻¹ := by rw [neg_inv]
                                      /-
                                        🎉 no goals
                                      -/


                                            /-
                                              K : Type u_1
                                              inst✝¹ : DivisionMonoid K
                                              inst✝ : HasDistribNeg K
                                              ⊢ Eq (Inv.inv (-1)) (-1)
                                            -/
theorem inv_neg_one : (-1 : K)⁻¹ = -1 := by rw [← neg_inv, inv_one]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                                             /-
                                                               K : Type u_1
                                                               inst✝ : DivisionRing K
                                                               a : K
                                                               h : Ne a 0
                                                               ⊢ Eq (HDiv.hDiv a (Neg.neg a)) (-1)
                                                             -/
theorem div_neg_self {a : K} (h : a ≠ 0) : a / -a = -1 := by rw [div_neg_eq_neg_div, div_self h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                             /-
                                                               K : Type u_1
                                                               inst✝ : DivisionRing K
                                                               a : K
                                                               h : Ne a 0
                                                               ⊢ Eq (HDiv.hDiv (Neg.neg a) a) (-1)
                                                             -/
theorem neg_div_self {a : K} (h : a ≠ 0) : -a / a = -1 := by rw [neg_div, div_self h]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem div_sub_div_same (a b c : K) : a / c - b / c = (a - b) / c := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b c : K
    ⊢ Eq (HSub.hSub (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv (HSub.hSub a b) c)
  -/
  rw [sub_eq_add_neg, ← neg_div, div_add_div_same, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem same_sub_div {a b : K} (h : b ≠ 0) : (b - a) / b = 1 - a / b := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b : K
    h : Ne b 0
    ⊢ Eq (HDiv.hDiv (HSub.hSub b a) b) (HSub.hSub 1 (HDiv.hDiv a b))
  -/
  simpa only [← @div_self _ _ b h] using (div_sub_div_same b a b).symm
  /-
    🎉 no goals
  -/


theorem one_sub_div {a b : K} (h : b ≠ 0) : 1 - a / b = (b - a) / b :=
  (same_sub_div h).symm


theorem div_sub_same {a b : K} (h : b ≠ 0) : (a - b) / b = a / b - 1 := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b : K
    h : Ne b 0
    ⊢ Eq (HDiv.hDiv (HSub.hSub a b) b) (HSub.hSub (HDiv.hDiv a b) 1)
  -/
  simpa only [← @div_self _ _ b h] using (div_sub_div_same a b b).symm
  /-
    🎉 no goals
  -/


theorem div_sub_one {a b : K} (h : b ≠ 0) : a / b - 1 = (a - b) / b :=
  (div_sub_same h).symm


theorem sub_div (a b c : K) : (a - b) / c = a / c - b / c :=
  (div_sub_div_same _ _ _).symm


/-- See `inv_sub_inv` for the more convenient version when `K` is commutative. -/
theorem inv_sub_inv' {a b : K} (ha : a ≠ 0) (hb : b ≠ 0) : a⁻¹ - b⁻¹ = a⁻¹ * (b - a) * b⁻¹ :=
  let _ := invertibleOfNonzero ha; let _ := invertibleOfNonzero hb; invOf_sub_invOf a b


theorem one_div_mul_sub_mul_one_div_eq_one_div_add_one_div (ha : a ≠ 0) (hb : b ≠ 0) :
    1 / a * (b - a) * (1 / b) = 1 / a - 1 / b := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b : K
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv 1 a) (HSub.hSub b a)) (HDiv.hDiv 1 b)) ( …
  -/
  simpa only [one_div] using (inv_sub_inv' ha hb).symm
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) DivisionRing.isDomain : IsDomain K :=
  NoZeroDivisors.to_isDomain _


protected theorem Commute.div_sub_div (hbc : Commute b c) (hbd : Commute b d) (hb : b ≠ 0)
    (hd : d ≠ 0) : a / b - c / d = (a * d - b * c) / (b * d) := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b c d : K
    hbc : Commute b c
    hbd : Commute b d
    hb : Ne b 0
    hd : Ne d 0
    ⊢ Eq (HSub.hSub (HDiv.hDiv a b) (HDiv.hDiv c d)) (HDiv.hDiv (HSub.hSub (HMul.h …
  -/
  simpa only [mul_neg, neg_div, ← sub_eq_add_neg] using hbc.neg_right.div_add_div hbd hb hd
  /-
    🎉 no goals
  -/


protected theorem Commute.inv_sub_inv (hab : Commute a b) (ha : a ≠ 0) (hb : b ≠ 0) :
    a⁻¹ - b⁻¹ = (b - a) / (a * b) := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    a b : K
    hab : Commute a b
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HSub.hSub (Inv.inv a) (Inv.inv b)) (HDiv.hDiv (HSub.hSub b a) (HMul.hMul …
  -/
  simp only [inv_eq_one_div, (Commute.one_right a).div_sub_div hab ha hb, one_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem div_add_div (a : K) (c : K) (hb : b ≠ 0) (hd : d ≠ 0) :
    a / b + c / d = (a * d + b * c) / (b * d) :=
  (Commute.all b _).div_add_div (Commute.all _ _) hb hd


theorem one_div_add_one_div (ha : a ≠ 0) (hb : b ≠ 0) : 1 / a + 1 / b = (a + b) / (a * b) :=
  (Commute.all a _).one_div_add_one_div ha hb


theorem inv_add_inv (ha : a ≠ 0) (hb : b ≠ 0) : a⁻¹ + b⁻¹ = (a + b) / (a * b) :=
  (Commute.all a _).inv_add_inv ha hb


@[field_simps]
theorem div_sub_div (a : K) {b : K} (c : K) {d : K} (hb : b ≠ 0) (hd : d ≠ 0) :
    a / b - c / d = (a * d - b * c) / (b * d) :=
  (Commute.all b _).div_sub_div (Commute.all _ _) hb hd


theorem inv_sub_inv {a b : K} (ha : a ≠ 0) (hb : b ≠ 0) : a⁻¹ - b⁻¹ = (b - a) / (a * b) := by
  /-
    K : Type u_1
    inst✝ : Field K
    a b : K
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HSub.hSub (Inv.inv a) (Inv.inv b)) (HDiv.hDiv (HSub.hSub b a) (HMul.hMul …
  -/
  rw [inv_eq_one_div, inv_eq_one_div, div_sub_div _ _ ha hb, one_mul, mul_one]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem sub_div' (a b c : K) (hc : c ≠ 0) : b - a / c = (b * c - a) / c := by
  /-
    K : Type u_1
    inst✝ : Field K
    a b c : K
    hc : Ne c 0
    ⊢ Eq (HSub.hSub b (HDiv.hDiv a c)) (HDiv.hDiv (HSub.hSub (HMul.hMul b c) a) c)
  -/
  simpa using div_sub_div b a one_ne_zero hc
  /-
    🎉 no goals
  -/


@[field_simps]
theorem div_sub' (a b c : K) (hc : c ≠ 0) : a / c - b = (a - c * b) / c := by
  /-
    K : Type u_1
    inst✝ : Field K
    a b c : K
    hc : Ne c 0
    ⊢ Eq (HSub.hSub (HDiv.hDiv a c) b) (HDiv.hDiv (HSub.hSub a (HMul.hMul c b)) c)
  -/
  simpa using div_sub_div a b hc one_ne_zero
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) Field.isDomain : IsDomain K :=
  { DivisionRing.isDomain with }


/-- Constructs a `DivisionRing` structure on a `Ring` consisting only of units and 0. -/
-- See note [reducible non-instances]
noncomputable abbrev DivisionRing.ofIsUnitOrEqZero [Ring R] (h : ∀ a : R, IsUnit a ∨ a = 0) :
    DivisionRing R where
  toRing := ‹Ring R›
  __ := groupWithZeroOfIsUnitOrEqZero h
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- Constructs a `Field` structure on a `CommRing` consisting only of units and 0. -/
-- See note [reducible non-instances]
noncomputable abbrev Field.ofIsUnitOrEqZero [CommRing R] (h : ∀ a : R, IsUnit a ∨ a = 0) :
    Field R where
  toCommRing := ‹CommRing R›
  __ := DivisionRing.ofIsUnitOrEqZero h


/-- Pullback a `DivisionSemiring` along an injective function. -/
-- See note [reducible non-instances]
protected abbrev divisionSemiring [DivisionSemiring L] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (inv : ∀ x, f x⁻¹ = (f x)⁻¹) (div : ∀ x y, f (x / y) = f x / f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (nnqsmul : ∀ (q : ℚ≥0) (x), f (q • x) = q • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (nnratCast : ∀ q : ℚ≥0, f q = q) : DivisionSemiring K where
  toSemiring := hf.semiring f zero one add mul nsmul npow natCast
  __ := hf.groupWithZero f zero one mul inv div npow zpow
                              /-
                                K : Type u_1
                                L : Type u_2
                                inst✝¹⁸ : Zero K
                                inst✝¹⁷ : Add K
                                inst✝¹⁶ : Neg K
                                inst✝¹⁵ : Sub K
                                inst✝¹⁴ : One K
                                inst✝¹³ : Mul K
                                inst✝¹² : Inv K
                                inst✝¹¹ : Div K
                                inst✝¹⁰ : SMul Nat K
                                inst✝⁹ : SMul Int K
                                inst✝⁸ : SMul NNRat K
                                inst✝⁷ : SMul Rat K
                                inst✝⁶ : Pow K Nat
                                inst✝⁵ : Pow K Int
                                inst✝⁴ : NatCast K
                                inst✝³ : IntCast K
                                inst✝² : NNRatCast K
                                inst✝¹ : RatCast K
                                f : K → L
                                hf : Function.Injective f
                                inst✝ : DivisionSemiring L
                                zero : Eq (f 0) 0
                                one : Eq (f 1) 1
                                add : ∀ (x y : K), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                mul : ∀ (x y : K), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                inv : ∀ (x : K), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                div : ∀ (x y : K), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                nsmul : ∀ (n : Nat) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                nnqsmul : ∀ (q : NNRat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                                npow : ∀ (x : K) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                zpow : ∀ (x : K) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                nnratCast : ∀ (q : NNRat), Eq (f ↑q) ↑q
                                q : NNRat
                                ⊢ Eq (f ↑q) (f (HDiv.hDiv ↑q.num ↑q.den))
                              -/
  nnratCast_def q := hf <| by rw [nnratCast, NNRat.cast_def, div, natCast, natCast]
                              /-
                                🎉 no goals
                              -/
  nnqsmul := (· • ·)
                              /-
                                K : Type u_1
                                L : Type u_2
                                inst✝¹⁸ : Zero K
                                inst✝¹⁷ : Add K
                                inst✝¹⁶ : Neg K
                                inst✝¹⁵ : Sub K
                                inst✝¹⁴ : One K
                                inst✝¹³ : Mul K
                                inst✝¹² : Inv K
                                inst✝¹¹ : Div K
                                inst✝¹⁰ : SMul Nat K
                                inst✝⁹ : SMul Int K
                                inst✝⁸ : SMul NNRat K
                                inst✝⁷ : SMul Rat K
                                inst✝⁶ : Pow K Nat
                                inst✝⁵ : Pow K Int
                                inst✝⁴ : NatCast K
                                inst✝³ : IntCast K
                                inst✝² : NNRatCast K
                                inst✝¹ : RatCast K
                                f : K → L
                                hf : Function.Injective f
                                inst✝ : DivisionSemiring L
                                zero : Eq (f 0) 0
                                one : Eq (f 1) 1
                                add : ∀ (x y : K), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                mul : ∀ (x y : K), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                inv : ∀ (x : K), Eq (f (Inv.inv x)) (Inv.inv (f x))
                                div : ∀ (x y : K), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                                nsmul : ∀ (n : Nat) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                                nnqsmul : ∀ (q : NNRat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                                npow : ∀ (x : K) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                zpow : ∀ (x : K) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                                natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                                nnratCast : ∀ (q : NNRat), Eq (f ↑q) ↑q
                                q : NNRat
                                a : K
                                ⊢ Eq (f ((fun x1 x2 => HSMul.hSMul x1 x2) q a)) (f (HMul.hMul (↑q) a))
                              -/
  nnqsmul_def q a := hf <| by rw [nnqsmul, NNRat.smul_def, mul, nnratCast]
                              /-
                                🎉 no goals
                              -/


/-- Pullback a `DivisionSemiring` along an injective function. -/
-- See note [reducible non-instances]
protected abbrev divisionRing [DivisionRing L] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (nnqsmul : ∀ (q : ℚ≥0) (x), f (q • x) = q • f x) (qsmul : ∀ (q : ℚ) (x), f (q • x) = q • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) (nnratCast : ∀ q : ℚ≥0, f q = q)
    (ratCast : ∀ q : ℚ, f q = q) : DivisionRing K where
  toRing := hf.ring f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.groupWithZero f zero one mul inv div npow zpow
  __ := hf.divisionSemiring f zero one add mul inv div nsmul nnqsmul npow zpow natCast nnratCast
                            /-
                              K : Type u_1
                              L : Type u_2
                              inst✝¹⁸ : Zero K
                              inst✝¹⁷ : Add K
                              inst✝¹⁶ : Neg K
                              inst✝¹⁵ : Sub K
                              inst✝¹⁴ : One K
                              inst✝¹³ : Mul K
                              inst✝¹² : Inv K
                              inst✝¹¹ : Div K
                              inst✝¹⁰ : SMul Nat K
                              inst✝⁹ : SMul Int K
                              inst✝⁸ : SMul NNRat K
                              inst✝⁷ : SMul Rat K
                              inst✝⁶ : Pow K Nat
                              inst✝⁵ : Pow K Int
                              inst✝⁴ : NatCast K
                              inst✝³ : IntCast K
                              inst✝² : NNRatCast K
                              inst✝¹ : RatCast K
                              f : K → L
                              hf : Function.Injective f
                              inst✝ : DivisionRing L
                              zero : Eq (f 0) 0
                              one : Eq (f 1) 1
                              add : ∀ (x y : K), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                              mul : ∀ (x y : K), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                              neg : ∀ (x : K), Eq (f (Neg.neg x)) (Neg.neg (f x))
                              sub : ∀ (x y : K), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                              inv : ∀ (x : K), Eq (f (Inv.inv x)) (Inv.inv (f x))
                              div : ∀ (x y : K), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                              nsmul : ∀ (n : Nat) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                              zsmul : ∀ (n : Int) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                              nnqsmul : ∀ (q : NNRat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                              qsmul : ∀ (q : Rat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                              npow : ∀ (x : K) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                              zpow : ∀ (x : K) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                              natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                              intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                              nnratCast : ∀ (q : NNRat), Eq (f ↑q) ↑q
                              ratCast : ∀ (q : Rat), Eq (f ↑q) ↑q
                              q : Rat
                              ⊢ Eq (f ↑q) (f (HDiv.hDiv ↑q.num ↑q.den))
                            -/
  ratCast_def q := hf <| by rw [ratCast, div, intCast, natCast, Rat.cast_def]
                            /-
                              🎉 no goals
                            -/
  qsmul := (· • ·)
                            /-
                              K : Type u_1
                              L : Type u_2
                              inst✝¹⁸ : Zero K
                              inst✝¹⁷ : Add K
                              inst✝¹⁶ : Neg K
                              inst✝¹⁵ : Sub K
                              inst✝¹⁴ : One K
                              inst✝¹³ : Mul K
                              inst✝¹² : Inv K
                              inst✝¹¹ : Div K
                              inst✝¹⁰ : SMul Nat K
                              inst✝⁹ : SMul Int K
                              inst✝⁸ : SMul NNRat K
                              inst✝⁷ : SMul Rat K
                              inst✝⁶ : Pow K Nat
                              inst✝⁵ : Pow K Int
                              inst✝⁴ : NatCast K
                              inst✝³ : IntCast K
                              inst✝² : NNRatCast K
                              inst✝¹ : RatCast K
                              f : K → L
                              hf : Function.Injective f
                              inst✝ : DivisionRing L
                              zero : Eq (f 0) 0
                              one : Eq (f 1) 1
                              add : ∀ (x y : K), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                              mul : ∀ (x y : K), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                              neg : ∀ (x : K), Eq (f (Neg.neg x)) (Neg.neg (f x))
                              sub : ∀ (x y : K), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                              inv : ∀ (x : K), Eq (f (Inv.inv x)) (Inv.inv (f x))
                              div : ∀ (x y : K), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                              nsmul : ∀ (n : Nat) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                              zsmul : ∀ (n : Int) (x : K), Eq (f (HSMul.hSMul n x)) (HSMul.hSMul n (f x))
                              nnqsmul : ∀ (q : NNRat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                              qsmul : ∀ (q : Rat) (x : K), Eq (f (HSMul.hSMul q x)) (HSMul.hSMul q (f x))
                              npow : ∀ (x : K) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                              zpow : ∀ (x : K) (n : Int), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                              natCast : ∀ (n : Nat), Eq (f ↑n) ↑n
                              intCast : ∀ (n : Int), Eq (f ↑n) ↑n
                              nnratCast : ∀ (q : NNRat), Eq (f ↑q) ↑q
                              ratCast : ∀ (q : Rat), Eq (f ↑q) ↑q
                              q : Rat
                              a : K
                              ⊢ Eq (f ((fun x1 x2 => HSMul.hSMul x1 x2) q a)) (f (HMul.hMul (↑q) a))
                            -/
  qsmul_def q a := hf <| by rw [qsmul, mul, Rat.smul_def, ratCast]
                            /-
                              🎉 no goals
                            -/


/-- Pullback a `Field` along an injective function. -/
-- See note [reducible non-instances]
protected abbrev semifield [Semifield L] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (inv : ∀ x, f x⁻¹ = (f x)⁻¹) (div : ∀ x y, f (x / y) = f x / f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (nnqsmul : ∀ (q : ℚ≥0) (x), f (q • x) = q • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (nnratCast : ∀ q : ℚ≥0, f q = q) : Semifield K where
  toCommSemiring := hf.commSemiring f zero one add mul nsmul npow natCast
  __ := hf.commGroupWithZero f zero one mul inv div npow zpow
  __ := hf.divisionSemiring f zero one add mul inv div nsmul nnqsmul npow zpow natCast nnratCast


/-- Pullback a `Field` along an injective function. -/
-- See note [reducible non-instances]
protected abbrev field [Field L] (zero : f 0 = 0) (one : f 1 = 1)
    (add : ∀ x y, f (x + y) = f x + f y) (mul : ∀ x y, f (x * y) = f x * f y)
    (neg : ∀ x, f (-x) = -f x) (sub : ∀ x y, f (x - y) = f x - f y) (inv : ∀ x, f x⁻¹ = (f x)⁻¹)
    (div : ∀ x y, f (x / y) = f x / f y)
    (nsmul : ∀ (n : ℕ) (x), f (n • x) = n • f x) (zsmul : ∀ (n : ℤ) (x), f (n • x) = n • f x)
    (nnqsmul : ∀ (q : ℚ≥0) (x), f (q • x) = q • f x) (qsmul : ∀ (q : ℚ) (x), f (q • x) = q • f x)
    (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n) (zpow : ∀ (x) (n : ℤ), f (x ^ n) = f x ^ n)
    (natCast : ∀ n : ℕ, f n = n) (intCast : ∀ n : ℤ, f n = n) (nnratCast : ∀ q : ℚ≥0, f q = q)
    (ratCast : ∀ q : ℚ, f q = q) :
    Field K where
  toCommRing := hf.commRing f zero one add mul neg sub nsmul zsmul npow natCast intCast
  __ := hf.divisionRing f zero one add mul neg sub inv div nsmul zsmul nnqsmul qsmul npow zpow
    natCast intCast nnratCast ratCast


instance instRatCast [RatCast K] : RatCast Kᵒᵈ := ‹_›

instance instDivisionSemiring [DivisionSemiring K] : DivisionSemiring Kᵒᵈ := ‹_›

instance instDivisionRing [DivisionRing K] : DivisionRing Kᵒᵈ := ‹_›

instance instSemifield [Semifield K] : Semifield Kᵒᵈ := ‹_›

instance instField [Field K] : Field Kᵒᵈ := ‹_›


@[simp] lemma toDual_ratCast [RatCast K] (n : ℚ) : toDual (n : K) = n := rfl


@[simp] lemma ofDual_ratCast [RatCast K] (n : ℚ) : (ofDual n : K) = n := rfl


instance instRatCast [RatCast K] : RatCast (Lex K) := ‹_›

instance instDivisionSemiring [DivisionSemiring K] : DivisionSemiring (Lex K) := ‹_›

instance instDivisionRing [DivisionRing K] : DivisionRing (Lex K) := ‹_›

instance instSemifield [Semifield K] : Semifield (Lex K) := ‹_›

instance instField [Field K] : Field (Lex K) := ‹_›


@[simp] lemma toLex_ratCast [RatCast K] (n : ℚ) : toLex (n : K) = n := rfl


@[simp] lemma ofLex_ratCast [RatCast K] (n : ℚ) : (ofLex n : K) = n := rfl

