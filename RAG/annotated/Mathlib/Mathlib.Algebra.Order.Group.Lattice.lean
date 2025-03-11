@[to_additive]
lemma mul_sup [MulLeftMono α] (a b c : α) :
    c * (a ⊔ b) = c * a ⊔ c * b :=
  (OrderIso.mulLeft _).map_sup _ _


@[to_additive]
lemma sup_mul [MulRightMono α] (a b c : α) :
    (a ⊔ b) * c = a * c ⊔ b * c :=
  (OrderIso.mulRight _).map_sup _ _


@[to_additive]
lemma mul_inf [MulLeftMono α] (a b c : α) :
    c * (a ⊓ b) = c * a ⊓ c * b :=
  (OrderIso.mulLeft _).map_inf _ _


@[to_additive]
lemma inf_mul [MulRightMono α] (a b c : α) :
    (a ⊓ b) * c = a * c ⊓ b * c :=
  (OrderIso.mulRight _).map_inf _ _


@[to_additive]
lemma sup_div [MulRightMono α] (a b c : α) :
    (a ⊔ b) / c = a / c ⊔ b / c :=
  (OrderIso.divRight _).map_sup _ _


@[to_additive]
lemma inf_div [MulRightMono α] (a b c : α) :
    (a ⊓ b) / c = a / c ⊓ b / c :=
  (OrderIso.divRight _).map_inf _ _


@[to_additive] lemma inv_sup (a b : α) : (a ⊔ b)⁻¹ = a⁻¹ ⊓ b⁻¹ := (OrderIso.inv α).map_sup _ _


@[to_additive] lemma inv_inf (a b : α) : (a ⊓ b)⁻¹ = a⁻¹ ⊔ b⁻¹ := (OrderIso.inv α).map_inf _ _


@[to_additive]
lemma div_sup (a b c : α) : c / (a ⊔ b) = c / a ⊓ c / b := (OrderIso.divLeft c).map_sup _ _


@[to_additive]
lemma div_inf (a b c : α) : c / (a ⊓ b) = c / a ⊔ c / b := (OrderIso.divLeft c).map_inf _ _

-- In fact 0 ≤ n•a implies 0 ≤ a, see L. Fuchs, "Partially ordered algebraic systems"
-- Chapter V, 1.E
-- See also `one_le_pow_iff` for the existing version in linear orders

@[to_additive]
lemma pow_two_semiclosed
    {a : α} (ha : 1 ≤ a ^ 2) : 1 ≤ a := by
  suffices this : (a ⊓ 1) * (a ⊓ 1) = a ⊓ 1 by
    rwa [← inf_eq_right, ← mul_right_eq_self]
  rw [mul_inf, inf_mul, ← pow_two, mul_one, one_mul, inf_assoc, inf_left_idem, inf_comm,
    inf_assoc, inf_of_le_left ha]


@[to_additive]
lemma inf_mul_sup [MulLeftMono α] (a b : α) : (a ⊓ b) * (a ⊔ b) = a * b :=
  calc
    (a ⊓ b) * (a ⊔ b) = (a ⊓ b) * (a * b * (b⁻¹ ⊔ a⁻¹)) := by
      /-
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : CommGroup α
        inst✝ : MulLeftMono α
        a b : α
        ⊢ Eq (HMul.hMul (Min.min a b) (Max.max a b)) (HMul.hMul (Min.min a b) (HMul.hM …
      -/
      rw [mul_sup b⁻¹ a⁻¹ (a * b), mul_inv_cancel_right, mul_inv_cancel_comm]
      /-
        🎉 no goals
      -/
                                            /-
                                              α : Type u_1
                                              inst✝² : Lattice α
                                              inst✝¹ : CommGroup α
                                              inst✝ : MulLeftMono α
                                              a b : α
                                              ⊢ Eq (HMul.hMul (Min.min a b) (HMul.hMul (HMul.hMul a b) (Max.max (Inv.inv b)  …
                                            -/
    _ = (a ⊓ b) * (a * b * (a ⊓ b)⁻¹) := by rw [inv_inf, sup_comm]
                                            /-
                                              🎉 no goals
                                            -/
                    /-
                      α : Type u_1
                      inst✝² : Lattice α
                      inst✝¹ : CommGroup α
                      inst✝ : MulLeftMono α
                      a b : α
                      ⊢ Eq (HMul.hMul (Min.min a b) (HMul.hMul (HMul.hMul a b) (Inv.inv (Min.min a b …
                    -/
    _ = a * b := by rw [mul_comm, inv_mul_cancel_right]
                    /-
                      🎉 no goals
                    -/


/-- Every lattice ordered commutative group is a distributive lattice. -/
-- Non-comm case needs cancellation law https://ncatlab.org/nlab/show/distributive+lattice
@[to_additive "Every lattice ordered commutative additive group is a distributive lattice"]
def CommGroup.toDistribLattice (α : Type*) [Lattice α] [CommGroup α]
    [MulLeftMono α] : DistribLattice α where
  le_sup_inf x y z := by
    rw [← mul_le_mul_iff_left (x ⊓ (y ⊓ z)), inf_mul_sup x (y ⊓ z), ← inv_mul_le_iff_le_mul,
      le_inf_iff]
    /-
      α✝ : Type u_1
      inst✝⁴ : Lattice α✝
      inst✝³ : CommGroup α✝
      α : Type u_2
      inst✝² : Lattice α
      inst✝¹ : CommGroup α
      inst✝ : MulLeftMono α
      x y z : α
      ⊢ And (LE.le (HMul.hMul (Inv.inv x) (HMul.hMul (Min.min x (Min.min y z)) (Min. …
    -/
    constructor
      /-
        case left
        α✝ : Type u_1
        inst✝⁴ : Lattice α✝
        inst✝³ : CommGroup α✝
        α : Type u_2
        inst✝² : Lattice α
        inst✝¹ : CommGroup α
        inst✝ : MulLeftMono α
        x y z : α
        ⊢ LE.le (HMul.hMul (Inv.inv x) (HMul.hMul (Min.min x (Min.min y z)) (Min.min ( …
      -/
    · rw [inv_mul_le_iff_le_mul, ← inf_mul_sup x y]
      /-
        case left
        α✝ : Type u_1
        inst✝⁴ : Lattice α✝
        inst✝³ : CommGroup α✝
        α : Type u_2
        inst✝² : Lattice α
        inst✝¹ : CommGroup α
        inst✝ : MulLeftMono α
        x y z : α
        ⊢ LE.le (HMul.hMul (Min.min x (Min.min y z)) (Min.min (Max.max x y) (Max.max x …
      -/
      exact mul_le_mul' (inf_le_inf_left _ inf_le_left) inf_le_left
      /-
        🎉 no goals
      -/
      /-
        case right
        α✝ : Type u_1
        inst✝⁴ : Lattice α✝
        inst✝³ : CommGroup α✝
        α : Type u_2
        inst✝² : Lattice α
        inst✝¹ : CommGroup α
        inst✝ : MulLeftMono α
        x y z : α
        ⊢ LE.le (HMul.hMul (Inv.inv x) (HMul.hMul (Min.min x (Min.min y z)) (Min.min ( …
      -/
    · rw [inv_mul_le_iff_le_mul, ← inf_mul_sup x z]
      /-
        case right
        α✝ : Type u_1
        inst✝⁴ : Lattice α✝
        inst✝³ : CommGroup α✝
        α : Type u_2
        inst✝² : Lattice α
        inst✝¹ : CommGroup α
        inst✝ : MulLeftMono α
        x y z : α
        ⊢ LE.le (HMul.hMul (Min.min x (Min.min y z)) (Min.min (Max.max x y) (Max.max x …
      -/
      exact mul_le_mul' (inf_le_inf_left _ inf_le_right) inf_le_right
      /-
        🎉 no goals
      -/

