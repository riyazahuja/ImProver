theorem map_dvd_iff (f : F) {a b} : f a ∣ f b ↔ a ∣ b :=
  let f := MulEquivClass.toMulEquiv f
              /-
                α : Type u_1
                β : Type u_2
                inst✝³ : Semigroup α
                inst✝² : Semigroup β
                F : Type u_3
                inst✝¹ : EquivLike F α β
                inst✝ : MulEquivClass F α β
                f✝ : F
                a b : α
                f : MulEquiv α β := ↑f✝
                h : Dvd.dvd (f✝ a) (f✝ b)
                ⊢ Dvd.dvd a b
              -/
  ⟨fun h ↦ by rw [← f.left_inv a, ← f.left_inv b]; exact map_dvd f.symm h, map_dvd f⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem MulEquiv.decompositionMonoid (f : F) [DecompositionMonoid β] : DecompositionMonoid α where
  primal a b c h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h : Dvd.dvd a (HMul.hMul b c)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
    rw [← map_dvd_iff f, map_mul] at h
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h : Dvd.dvd (f a) (HMul.hMul (f b) (f c))
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
    obtain ⟨a₁, a₂, h⟩ := DecompositionMonoid.primal _ h
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h✝ : Dvd.dvd (f a) (HMul.hMul (f b) (f c))
      a₁ a₂ : β
      h : And (Dvd.dvd a₁ (f b)) (And (Dvd.dvd a₂ (f c)) (Eq (f a) (HMul.hMul a₁ a₂)))
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
    refine ⟨symm f a₁, symm f a₂, ?_⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h✝ : Dvd.dvd (f a) (HMul.hMul (f b) (f c))
      a₁ a₂ : β
      h : And (Dvd.dvd a₁ (f b)) (And (Dvd.dvd a₂ (f c)) (Eq (f a) (HMul.hMul a₁ a₂)))
      ⊢ And (Dvd.dvd ((↑f).symm a₁) b) (And (Dvd.dvd ((↑f).symm a₂) c) (Eq a (HMul.h …
    -/
    simp_rw [← map_dvd_iff f, ← map_mul, eq_symm_apply]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h✝ : Dvd.dvd (f a) (HMul.hMul (f b) (f c))
      a₁ a₂ : β
      h : And (Dvd.dvd a₁ (f b)) (And (Dvd.dvd a₂ (f c)) (Eq (f a) (HMul.hMul a₁ a₂)))
      ⊢ And (Dvd.dvd (f ((↑f).symm a₁)) (f b)) (And (Dvd.dvd (f ((↑f).symm a₂)) (f c …
    -/
    iterate 2 erw [(f : α ≃* β).apply_symm_apply]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Semigroup α
      inst✝³ : Semigroup β
      F : Type u_3
      inst✝² : EquivLike F α β
      inst✝¹ : MulEquivClass F α β
      f : F
      inst✝ : DecompositionMonoid β
      a b c : α
      h✝ : Dvd.dvd (f a) (HMul.hMul (f b) (f c))
      a₁ a₂ : β
      h : And (Dvd.dvd a₁ (f b)) (And (Dvd.dvd a₂ (f c)) (Eq (f a) (HMul.hMul a₁ a₂)))
      ⊢ And (Dvd.dvd a₁ (f b)) (And (Dvd.dvd a₂ (f c)) (Eq (↑f a) (HMul.hMul a₁ a₂)))
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem dvd_add [LeftDistribClass α] {a b c : α} (h₁ : a ∣ b) (h₂ : a ∣ c) : a ∣ b + c :=
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝² : Add α
                                                                          inst✝¹ : Semigroup α
                                                                          inst✝ : LeftDistribClass α
                                                                          a b c : α
                                                                          h₁ : Dvd.dvd a b
                                                                          h₂ : Dvd.dvd a c
                                                                          d : α
                                                                          hd : Eq b (HMul.hMul a d)
                                                                          e : α
                                                                          he : Eq c (HMul.hMul a e)
                                                                          ⊢ Eq (HMul.hMul a (HAdd.hAdd d e)) (HAdd.hAdd b c)
                                                                        -/
  Dvd.elim h₁ fun d hd => Dvd.elim h₂ fun e he => Dvd.intro (d + e) (by simp [left_distrib, hd, he])
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


alias Dvd.dvd.add := dvd_add


lemma min_pow_dvd_add (ha : c ^ m ∣ a) (hb : c ^ n ∣ b) : c ^ min m n ∣ a + b :=
  ((pow_dvd_pow c (m.min_le_left n)).trans ha).add ((pow_dvd_pow c (m.min_le_right n)).trans hb)


theorem Dvd.dvd.linear_comb {d x y : α} (hdx : d ∣ x) (hdy : d ∣ y) (a b : α) : d ∣ a * x + b * y :=
  dvd_add (hdx.mul_left a) (hdy.mul_left b)


/-- An element `a` of a semigroup with a distributive negation divides the negation of an element
`b` iff `a` divides `b`. -/
@[simp]
theorem dvd_neg : a ∣ -b ↔ a ∣ b :=
  (Equiv.neg _).exists_congr_left.trans <| by
    /-
      α : Type u_1
      inst✝¹ : Semigroup α
      inst✝ : HasDistribNeg α
      a b : α
      ⊢ Iff (Exists fun b_1 => Eq (Neg.neg b) (HMul.hMul a ((Equiv.symm (Equiv.neg α …
    -/
    simp only [Equiv.neg_symm, Equiv.neg_apply, mul_neg, neg_inj, Dvd.dvd]
    /-
      🎉 no goals
    -/


/-- The negation of an element `a` of a semigroup with a distributive negation divides another
element `b` iff `a` divides `b`. -/
@[simp]
theorem neg_dvd : -a ∣ b ↔ a ∣ b :=
  (Equiv.neg _).exists_congr_left.trans <| by
    /-
      α : Type u_1
      inst✝¹ : Semigroup α
      inst✝ : HasDistribNeg α
      a b : α
      ⊢ Iff (Exists fun b_1 => Eq b (HMul.hMul (Neg.neg a) ((Equiv.symm (Equiv.neg α …
    -/
    simp only [Equiv.neg_symm, Equiv.neg_apply, mul_neg, neg_mul, neg_neg, Dvd.dvd]
    /-
      🎉 no goals
    -/


alias ⟨Dvd.dvd.of_neg_left, Dvd.dvd.neg_left⟩ := neg_dvd


alias ⟨Dvd.dvd.of_neg_right, Dvd.dvd.neg_right⟩ := dvd_neg


theorem dvd_sub (h₁ : a ∣ b) (h₂ : a ∣ c) : a ∣ b - c := by
  /-
    α : Type u_1
    inst✝ : NonUnitalRing α
    a b c : α
    h₁ : Dvd.dvd a b
    h₂ : Dvd.dvd a c
    ⊢ Dvd.dvd a (HSub.hSub b c)
  -/
  simpa only [← sub_eq_add_neg] using h₁.add h₂.neg_right
  /-
    🎉 no goals
  -/


alias Dvd.dvd.sub := dvd_sub


/-- If an element `a` divides another element `c` in a ring, `a` divides the sum of another element
`b` with `c` iff `a` divides `b`. -/
theorem dvd_add_left (h : a ∣ c) : a ∣ b + c ↔ a ∣ b :=
               /-
                 α : Type u_1
                 inst✝ : NonUnitalRing α
                 a b c : α
                 h : Dvd.dvd a c
                 H : Dvd.dvd a (HAdd.hAdd b c)
                 ⊢ Dvd.dvd a b
               -/
  ⟨fun H => by simpa only [add_sub_cancel_right] using dvd_sub H h, fun h₂ => dvd_add h₂ h⟩
               /-
                 🎉 no goals
               -/


/-- If an element `a` divides another element `b` in a ring, `a` divides the sum of `b` and another
element `c` iff `a` divides `c`. -/
                                                            /-
                                                              α : Type u_1
                                                              inst✝ : NonUnitalRing α
                                                              a b c : α
                                                              h : Dvd.dvd a b
                                                              ⊢ Iff (Dvd.dvd a (HAdd.hAdd b c)) (Dvd.dvd a c)
                                                            -/
theorem dvd_add_right (h : a ∣ b) : a ∣ b + c ↔ a ∣ c := by rw [add_comm]; exact dvd_add_left h
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- If an element `a` divides another element `c` in a ring, `a` divides the difference of another
element `b` with `c` iff `a` divides `b`. -/
theorem dvd_sub_left (h : a ∣ c) : a ∣ b - c ↔ a ∣ b := by
  -- Porting note: Needed to give `α` explicitly
  /-
    α : Type u_1
    inst✝ : NonUnitalRing α
    a b c : α
    h : Dvd.dvd a c
    ⊢ Iff (Dvd.dvd a (HSub.hSub b c)) (Dvd.dvd a b)
  -/
  simpa only [← sub_eq_add_neg] using dvd_add_left ((dvd_neg (α := α)).2 h)
  /-
    🎉 no goals
  -/


/-- If an element `a` divides another element `b` in a ring, `a` divides the difference of `b` and
another element `c` iff `a` divides `c`. -/
theorem dvd_sub_right (h : a ∣ b) : a ∣ b - c ↔ a ∣ c := by
  -- Porting note: Needed to give `α` explicitly
  /-
    α : Type u_1
    inst✝ : NonUnitalRing α
    a b c : α
    h : Dvd.dvd a b
    ⊢ Iff (Dvd.dvd a (HSub.hSub b c)) (Dvd.dvd a c)
  -/
  rw [sub_eq_add_neg, dvd_add_right h, dvd_neg (α := α)]
  /-
    🎉 no goals
  -/


theorem dvd_iff_dvd_of_dvd_sub (h : a ∣ b - c) : a ∣ b ↔ a ∣ c := by
  /-
    α : Type u_1
    inst✝ : NonUnitalRing α
    a b c : α
    h : Dvd.dvd a (HSub.hSub b c)
    ⊢ Iff (Dvd.dvd a b) (Dvd.dvd a c)
  -/
  rw [← sub_add_cancel b c, dvd_add_right h]
  /-
    🎉 no goals
  -/

-- Porting note: Needed to give `α` explicitly

                                                   /-
                                                     α : Type u_1
                                                     inst✝ : NonUnitalRing α
                                                     a b c : α
                                                     ⊢ Iff (Dvd.dvd a (HSub.hSub b c)) (Dvd.dvd a (HSub.hSub c b))
                                                   -/
theorem dvd_sub_comm : a ∣ b - c ↔ a ∣ c - b := by rw [← dvd_neg (α := α), neg_sub]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- An element a divides the sum a + b if and only if a divides b. -/
@[simp]
theorem dvd_add_self_left {a b : α} : a ∣ a + b ↔ a ∣ b :=
  dvd_add_right (dvd_refl a)


/-- An element a divides the sum b + a if and only if a divides b. -/
@[simp]
theorem dvd_add_self_right {a b : α} : a ∣ b + a ↔ a ∣ b :=
  dvd_add_left (dvd_refl a)


/-- An element `a` divides the difference `a - b` if and only if `a` divides `b`. -/
@[simp]
theorem dvd_sub_self_left : a ∣ a - b ↔ a ∣ b :=
  dvd_sub_right dvd_rfl


/-- An element `a` divides the difference `b - a` if and only if `a` divides `b`. -/
@[simp]
theorem dvd_sub_self_right : a ∣ b - a ↔ a ∣ b :=
  dvd_sub_left dvd_rfl


theorem dvd_mul_sub_mul {k a b x y : α} (hab : k ∣ a - b) (hxy : k ∣ x - y) :
    k ∣ a * x - b * y := by
  /-
    α : Type u_1
    inst✝ : NonUnitalCommRing α
    k a b x y : α
    hab : Dvd.dvd k (HSub.hSub a b)
    hxy : Dvd.dvd k (HSub.hSub x y)
    ⊢ Dvd.dvd k (HSub.hSub (HMul.hMul a x) (HMul.hMul b y))
  -/
  convert dvd_add (hxy.mul_left a) (hab.mul_right y) using 1
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : NonUnitalCommRing α
    k a b x y : α
    hab : Dvd.dvd k (HSub.hSub a b)
    hxy : Dvd.dvd k (HSub.hSub x y)
    ⊢ Eq (HSub.hSub (HMul.hMul a x) (HMul.hMul b y)) (HAdd.hAdd (HMul.hMul a (HSub …
  -/
  rw [mul_sub_left_distrib, mul_sub_right_distrib]
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : NonUnitalCommRing α
    k a b x y : α
    hab : Dvd.dvd k (HSub.hSub a b)
    hxy : Dvd.dvd k (HSub.hSub x y)
    ⊢ Eq (HSub.hSub (HMul.hMul a x) (HMul.hMul b y)) (HAdd.hAdd (HSub.hSub (HMul.h …
  -/
  simp only [sub_eq_add_neg, add_assoc, neg_add_cancel_left]
  /-
    🎉 no goals
  -/


