@[simp]
theorem add_right [Distrib R] {a x y x' y' : R} (h : SemiconjBy a x y) (h' : SemiconjBy a x' y') :
    SemiconjBy a (x + x') (y + y') := by
  /-
    R : Type u
    inst✝ : Distrib R
    a x y x' y' : R
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ SemiconjBy a (HAdd.hAdd x x') (HAdd.hAdd y y')
  -/
  simp only [SemiconjBy, left_distrib, right_distrib, h.eq, h'.eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_left [Distrib R] {a b x y : R} (ha : SemiconjBy a x y) (hb : SemiconjBy b x y) :
    SemiconjBy (a + b) x y := by
  /-
    R : Type u
    inst✝ : Distrib R
    a b x y : R
    ha : SemiconjBy a x y
    hb : SemiconjBy b x y
    ⊢ SemiconjBy (HAdd.hAdd a b) x y
  -/
  simp only [SemiconjBy, left_distrib, right_distrib, ha.eq, hb.eq]
  /-
    🎉 no goals
  -/


theorem neg_right (h : SemiconjBy a x y) : SemiconjBy a (-x) (-y) := by
  /-
    R : Type u
    inst✝¹ : Mul R
    inst✝ : HasDistribNeg R
    a x y : R
    h : SemiconjBy a x y
    ⊢ SemiconjBy a (Neg.neg x) (Neg.neg y)
  -/
  simp only [SemiconjBy, h.eq, neg_mul, mul_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_right_iff : SemiconjBy a (-x) (-y) ↔ SemiconjBy a x y :=
  ⟨fun h => neg_neg x ▸ neg_neg y ▸ h.neg_right, SemiconjBy.neg_right⟩


theorem neg_left (h : SemiconjBy a x y) : SemiconjBy (-a) x y := by
  /-
    R : Type u
    inst✝¹ : Mul R
    inst✝ : HasDistribNeg R
    a x y : R
    h : SemiconjBy a x y
    ⊢ SemiconjBy (Neg.neg a) x y
  -/
  simp only [SemiconjBy, h.eq, neg_mul, mul_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_left_iff : SemiconjBy (-a) x y ↔ SemiconjBy a x y :=
  ⟨fun h => neg_neg a ▸ h.neg_left, SemiconjBy.neg_left⟩


                                                             /-
                                                               R : Type u
                                                               inst✝¹ : MulOneClass R
                                                               inst✝ : HasDistribNeg R
                                                               a : R
                                                               ⊢ SemiconjBy a (-1) (-1)
                                                             -/
theorem neg_one_right (a : R) : SemiconjBy a (-1) (-1) := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                         /-
                                                           R : Type u
                                                           inst✝¹ : MulOneClass R
                                                           inst✝ : HasDistribNeg R
                                                           x : R
                                                           ⊢ SemiconjBy (-1) x x
                                                         -/
theorem neg_one_left (x : R) : SemiconjBy (-1) x x := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem sub_right (h : SemiconjBy a x y) (h' : SemiconjBy a x' y') :
    SemiconjBy a (x - x') (y - y') := by
  /-
    R : Type u
    inst✝ : NonUnitalNonAssocRing R
    a x y x' y' : R
    h : SemiconjBy a x y
    h' : SemiconjBy a x' y'
    ⊢ SemiconjBy a (HSub.hSub x x') (HSub.hSub y y')
  -/
  simpa only [sub_eq_add_neg] using h.add_right h'.neg_right
  /-
    🎉 no goals
  -/


@[simp]
theorem sub_left (ha : SemiconjBy a x y) (hb : SemiconjBy b x y) :
    SemiconjBy (a - b) x y := by
  /-
    R : Type u
    inst✝ : NonUnitalNonAssocRing R
    a b x y : R
    ha : SemiconjBy a x y
    hb : SemiconjBy b x y
    ⊢ SemiconjBy (HSub.hSub a b) x y
  -/
  simpa only [sub_eq_add_neg] using ha.add_left hb.neg_left
  /-
    🎉 no goals
  -/


