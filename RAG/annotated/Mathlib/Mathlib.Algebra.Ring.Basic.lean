/-- Left multiplication by an element of a type with distributive multiplication is an `AddHom`. -/
@[simps (config := .asFn)]
def mulLeft [Distrib R] (r : R) : AddHom R R where
  toFun := (r * ·)
  map_add' := mul_add r


/-- Left multiplication by an element of a type with distributive multiplication is an `AddHom`. -/
@[simps (config := .asFn)]
def mulRight [Distrib R] (r : R) : AddHom R R where
  toFun a := a * r
  map_add' _ _ := add_mul _ _ r


/-- Left multiplication by an element of a (semi)ring is an `AddMonoidHom` -/
def mulLeft [NonUnitalNonAssocSemiring R] (r : R) : R →+ R where
  toFun := (r * ·)
  map_zero' := mul_zero r
  map_add' := mul_add r


@[simp]
theorem coe_mulLeft [NonUnitalNonAssocSemiring R] (r : R) :
    (mulLeft r : R → R) = HMul.hMul r :=
  rfl


/-- Right multiplication by an element of a (semi)ring is an `AddMonoidHom` -/
def mulRight [NonUnitalNonAssocSemiring R] (r : R) : R →+ R where
  toFun a := a * r
  map_zero' := zero_mul r
  map_add' _ _ := add_mul _ _ r


@[simp]
theorem coe_mulRight [NonUnitalNonAssocSemiring R] (r : R) :
    (mulRight r) = (· * r) :=
  rfl


theorem mulRight_apply [NonUnitalNonAssocSemiring R] (a r : R) :
    mulRight r a = a * r :=
  rfl


instance instHasDistribNeg : HasDistribNeg αᵐᵒᵖ where
  neg_mul _ _ := unop_injective <| mul_neg _ _
  mul_neg _ _ := unop_injective <| neg_mul _ _


@[simp]
theorem inv_neg' (a : α) : (-a)⁻¹ = -a⁻¹ := by
  /-
    α : Type u_2
    inst✝¹ : Group α
    inst✝ : HasDistribNeg α
    a : α
    ⊢ Eq (Inv.inv (Neg.neg a)) (Neg.neg (Inv.inv a))
  -/
  rw [eq_comm, eq_inv_iff_mul_eq_one, neg_mul, mul_neg, neg_neg, inv_mul_cancel]
  /-
    🎉 no goals
  -/


/-- Vieta's formula for a quadratic equation, relating the coefficients of the polynomial with
  its roots. This particular version states that if we have a root `x` of a monic quadratic
  polynomial, then there is another root `y` such that `x + y` is negative the `a_1` coefficient
  and `x * y` is the `a_0` coefficient. -/
theorem vieta_formula_quadratic {b c x : α} (h : x * x - b * x + c = 0) :
    ∃ y : α, y * y - b * y + c = 0 ∧ x + y = b ∧ x * y = c := by
  /-
    α : Type u_2
    inst✝ : NonUnitalCommRing α
    b c x : α
    h : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul x x) (HMul.hMul b x)) c) 0
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HSub.hSub (HMul.hMul y y) (HMul.hMul b y …
  -/
  have : c = x * (b - x) := (eq_neg_of_add_eq_zero_right h).trans (by simp [mul_sub, mul_comm])
  /-
    α : Type u_2
    inst✝ : NonUnitalCommRing α
    b c x : α
    h : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul x x) (HMul.hMul b x)) c) 0
    this : Eq c (HMul.hMul x (HSub.hSub b x))
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HSub.hSub (HMul.hMul y y) (HMul.hMul b y …
  -/
  refine ⟨b - x, ?_, by simp, by rw [this]⟩
  /-
    α : Type u_2
    inst✝ : NonUnitalCommRing α
    b c x : α
    h : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul x x) (HMul.hMul b x)) c) 0
    this : Eq c (HMul.hMul x (HSub.hSub b x))
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (HSub.hSub b x) (HSub.hSub b x)) (HMul.h …
  -/
  rw [this, sub_add, ← sub_mul, sub_self]
  /-
    🎉 no goals
  -/


theorem succ_ne_self {α : Type*} [NonAssocRing α] [Nontrivial α] (a : α) : a + 1 ≠ a := fun h =>
                                        /-
                                          α : Type u_2
                                          inst✝¹ : NonAssocRing α
                                          inst✝ : Nontrivial α
                                          a : α
                                          h : Eq (HAdd.hAdd a 1) a
                                          ⊢ Eq (HAdd.hAdd a 1) (HAdd.hAdd a 0)
                                        -/
  one_ne_zero ((add_right_inj a).mp (by simp [h]))
                                        /-
                                          🎉 no goals
                                        -/


theorem pred_ne_self {α : Type*} [NonAssocRing α] [Nontrivial α] (a : α) : a - 1 ≠ a := fun h ↦
                                                       /-
                                                         α : Type u_2
                                                         inst✝¹ : NonAssocRing α
                                                         inst✝ : Nontrivial α
                                                         a : α
                                                         h : Eq (HSub.hSub a 1) a
                                                         ⊢ Eq (HAdd.hAdd a (-1)) (HAdd.hAdd a (-0))
                                                       -/
  one_ne_zero (neg_injective ((add_right_inj a).mp (by simp [← sub_eq_add_neg, h])))
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma IsLeftCancelMulZero.to_noZeroDivisors [NonUnitalNonAssocSemiring α]
    [IsLeftCancelMulZero α] : NoZeroDivisors α where
  eq_zero_or_eq_zero_of_mul_eq_zero {x _} h :=
    or_iff_not_imp_left.mpr fun ne ↦ mul_left_cancel₀ ne ((mul_zero x).symm ▸ h)


lemma IsRightCancelMulZero.to_noZeroDivisors [NonUnitalNonAssocSemiring α]
    [IsRightCancelMulZero α] : NoZeroDivisors α where
  eq_zero_or_eq_zero_of_mul_eq_zero {_ y} h :=
    or_iff_not_imp_right.mpr fun ne ↦ mul_right_cancel₀ ne ((zero_mul y).symm ▸ h)


instance (priority := 100) NoZeroDivisors.to_isCancelMulZero
    [NonUnitalNonAssocRing α] [NoZeroDivisors α] :
    IsCancelMulZero α where
  mul_left_cancel_of_ne_zero ha h := by
    /-
      R : Type u_1
      α : Type u_2
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      a✝ b✝ c✝ : α
      ha : Ne a✝ 0
      h : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
      ⊢ Eq b✝ c✝
    -/
    rw [← sub_eq_zero, ← mul_sub] at h
    /-
      R : Type u_1
      α : Type u_2
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      a✝ b✝ c✝ : α
      ha : Ne a✝ 0
      h : Eq (HMul.hMul a✝ (HSub.hSub b✝ c✝)) 0
      ⊢ Eq b✝ c✝
    -/
    exact sub_eq_zero.1 ((eq_zero_or_eq_zero_of_mul_eq_zero h).resolve_left ha)
    /-
      🎉 no goals
    -/
  mul_right_cancel_of_ne_zero hb h := by
    /-
      R : Type u_1
      α : Type u_2
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      a✝ b✝ c✝ : α
      hb : Ne b✝ 0
      h : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
      ⊢ Eq a✝ c✝
    -/
    rw [← sub_eq_zero, ← sub_mul] at h
    /-
      R : Type u_1
      α : Type u_2
      inst✝¹ : NonUnitalNonAssocRing α
      inst✝ : NoZeroDivisors α
      a✝ b✝ c✝ : α
      hb : Ne b✝ 0
      h : Eq (HMul.hMul (HSub.hSub a✝ c✝) b✝) 0
      ⊢ Eq a✝ c✝
    -/
    exact sub_eq_zero.1 ((eq_zero_or_eq_zero_of_mul_eq_zero h).resolve_right hb)
    /-
      🎉 no goals
    -/


/-- In a ring, `IsCancelMulZero` and `NoZeroDivisors` are equivalent. -/
lemma isCancelMulZero_iff_noZeroDivisors [NonUnitalNonAssocRing α] :
    IsCancelMulZero α ↔ NoZeroDivisors α :=
  ⟨fun _ => IsRightCancelMulZero.to_noZeroDivisors _, fun _ => inferInstance⟩


lemma NoZeroDivisors.to_isDomain [Ring α] [h : Nontrivial α] [NoZeroDivisors α] :
    IsDomain α :=
  { NoZeroDivisors.to_isCancelMulZero α, h with .. }


instance (priority := 100) IsDomain.to_noZeroDivisors [Semiring α] [IsDomain α] :
    NoZeroDivisors α :=
  IsRightCancelMulZero.to_noZeroDivisors α


instance Subsingleton.to_isCancelMulZero [Mul α] [Zero α] [Subsingleton α] : IsCancelMulZero α where
  mul_right_cancel_of_ne_zero hb := (hb <| Subsingleton.eq_zero _).elim
  mul_left_cancel_of_ne_zero hb := (hb <| Subsingleton.eq_zero _).elim


instance Subsingleton.to_noZeroDivisors [Mul α] [Zero α] [Subsingleton α] : NoZeroDivisors α where
  eq_zero_or_eq_zero_of_mul_eq_zero _ := .inl (Subsingleton.eq_zero _)


lemma isDomain_iff_cancelMulZero_and_nontrivial [Semiring α] :
    IsDomain α ↔ IsCancelMulZero α ∧ Nontrivial α :=
  ⟨fun _ => ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ => {}⟩


lemma isCancelMulZero_iff_isDomain_or_subsingleton [Semiring α] :
    IsCancelMulZero α ↔ IsDomain α ∨ Subsingleton α := by
  /-
    α : Type u_2
    inst✝ : Semiring α
    ⊢ Iff (IsCancelMulZero α) (Or (IsDomain α) (Subsingleton α))
  -/
  refine ⟨fun t ↦ ?_, fun h ↦ h.elim (fun _ ↦ inferInstance) (fun _ ↦ inferInstance)⟩
  /-
    α : Type u_2
    inst✝ : Semiring α
    t : IsCancelMulZero α
    ⊢ Or (IsDomain α) (Subsingleton α)
  -/
  rw [or_iff_not_imp_right, not_subsingleton_iff_nontrivial]
  /-
    α : Type u_2
    inst✝ : Semiring α
    t : IsCancelMulZero α
    ⊢ Nontrivial α → IsDomain α
  -/
  exact fun _ ↦ {}
  /-
    🎉 no goals
  -/


lemma isDomain_iff_noZeroDivisors_and_nontrivial [Ring α] :
    IsDomain α ↔ NoZeroDivisors α ∧ Nontrivial α := by
  /-
    α : Type u_2
    inst✝ : Ring α
    ⊢ Iff (IsDomain α) (And (NoZeroDivisors α) (Nontrivial α))
  -/
  rw [← isCancelMulZero_iff_noZeroDivisors, isDomain_iff_cancelMulZero_and_nontrivial]
  /-
    🎉 no goals
  -/


lemma noZeroDivisors_iff_isDomain_or_subsingleton [Ring α] :
    NoZeroDivisors α ↔ IsDomain α ∨ Subsingleton α := by
  /-
    α : Type u_2
    inst✝ : Ring α
    ⊢ Iff (NoZeroDivisors α) (Or (IsDomain α) (Subsingleton α))
  -/
  rw [← isCancelMulZero_iff_noZeroDivisors, isCancelMulZero_iff_isDomain_or_subsingleton]
  /-
    🎉 no goals
  -/


