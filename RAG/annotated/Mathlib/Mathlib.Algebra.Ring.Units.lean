/-- Each element of the group of units of a ring has an additive inverse. -/
instance : Neg αˣ :=
                            /-
                              α : Type u
                              β : Type v
                              R : Type x
                              inst✝¹ : Monoid α
                              inst✝ : HasDistribNeg α
                              u : Units α
                              ⊢ Eq (HMul.hMul (Neg.neg ↑u) (Neg.neg ↑(Inv.inv u))) 1
                            -/
                            /-
                              🎉 no goals
                            -/
  ⟨fun u => ⟨-↑u, -↑u⁻¹, by simp, by simp⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- Representing an element of a ring's unit group as an element of the ring commutes with
    mapping this element to its additive inverse. -/
@[simp, norm_cast]
protected theorem val_neg (u : αˣ) : (↑(-u) : α) = -u :=
  rfl


@[simp, norm_cast]
protected theorem coe_neg_one : ((-1 : αˣ) : α) = -1 :=
  rfl


instance : HasDistribNeg αˣ :=
  Units.ext.hasDistribNeg _ Units.val_neg Units.val_mul


@[field_simps]
                                                              /-
                                                                α : Type u
                                                                inst✝¹ : Monoid α
                                                                inst✝ : HasDistribNeg α
                                                                a : α
                                                                u : Units α
                                                                ⊢ Eq (Neg.neg (divp a u)) (divp (Neg.neg a) u)
                                                              -/
theorem neg_divp (a : α) (u : αˣ) : -(a /ₚ u) = -a /ₚ u := by simp only [divp, neg_mul]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[field_simps 1010]
theorem divp_add_divp_same (a b : α) (u : αˣ) : a /ₚ u + b /ₚ u = (a + b) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HAdd.hAdd (divp a u) (divp b u)) (divp (HAdd.hAdd a b) u)
  -/
  simp only [divp, add_mul]
  /-
    🎉 no goals
  -/

-- Needs to have higher simp priority than divp_sub_divp. 1000 is the default priority.

@[field_simps 1010]
theorem divp_sub_divp_same (a b : α) (u : αˣ) : a /ₚ u - b /ₚ u = (a - b) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HSub.hSub (divp a u) (divp b u)) (divp (HSub.hSub a b) u)
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, neg_divp, divp_add_divp_same]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem add_divp (a b : α) (u : αˣ) : a + b /ₚ u = (a * u + b) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HAdd.hAdd a (divp b u)) (divp (HAdd.hAdd (HMul.hMul a ↑u) b) u)
  -/
  simp only [divp, add_mul, Units.mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem sub_divp (a b : α) (u : αˣ) : a - b /ₚ u = (a * u - b) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HSub.hSub a (divp b u)) (divp (HSub.hSub (HMul.hMul a ↑u) b) u)
  -/
  simp only [divp, sub_mul, Units.mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem divp_add (a b : α) (u : αˣ) : a /ₚ u + b = (a + b * u) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HAdd.hAdd (divp a u) b) (divp (HAdd.hAdd a (HMul.hMul b ↑u)) u)
  -/
  simp only [divp, add_mul, Units.mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


@[field_simps]
theorem divp_sub (a b : α) (u : αˣ) : a /ₚ u - b = (a - b * u) /ₚ u := by
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq (HSub.hSub (divp a u) b) (divp (HSub.hSub a (HMul.hMul b ↑u)) u)
  -/
  simp only [divp, sub_mul, sub_right_inj]
  /-
    α : Type u
    inst✝ : Ring α
    a b : α
    u : Units α
    ⊢ Eq b (HMul.hMul (HMul.hMul b ↑u) ↑(Inv.inv u))
  -/
  rw [mul_assoc, Units.mul_inv, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem map_neg {F : Type*} [Ring β] [FunLike F α β] [RingHomClass F α β]
    (f : F) (u : αˣ) : map (f : α →* β) (-u) = -map (f : α →* β) u :=
          /-
            α : Type u
            β : Type v
            inst✝³ : Ring α
            F : Type u_1
            inst✝² : Ring β
            inst✝¹ : FunLike F α β
            inst✝ : RingHomClass F α β
            f : F
            u : Units α
            ⊢ Eq ↑((Units.map ↑f) (Neg.neg u)) ↑(Neg.neg ((Units.map ↑f) u))
          -/
  ext (by simp only [coe_map, Units.val_neg, MonoidHom.coe_coe, map_neg])
          /-
            🎉 no goals
          -/


protected theorem map_neg_one {F : Type*} [Ring β] [FunLike F α β] [RingHomClass F α β]
    (f : F) : map (f : α →* β) (-1) = -1 := by
  /-
    α : Type u
    β : Type v
    inst✝³ : Ring α
    F : Type u_1
    inst✝² : Ring β
    inst✝¹ : FunLike F α β
    inst✝ : RingHomClass F α β
    f : F
    ⊢ Eq ((Units.map ↑f) (-1)) (-1)
  -/
  simp only [Units.map_neg, map_one]
  /-
    🎉 no goals
  -/


theorem IsUnit.neg [Monoid α] [HasDistribNeg α] {a : α} : IsUnit a → IsUnit (-a)
  | ⟨x, hx⟩ => hx ▸ (-x).isUnit


@[simp]
theorem IsUnit.neg_iff [Monoid α] [HasDistribNeg α] (a : α) : IsUnit (-a) ↔ IsUnit a :=
  ⟨fun h => neg_neg a ▸ h.neg, IsUnit.neg⟩


theorem isUnit_neg_one [Monoid α] [HasDistribNeg α] : IsUnit (-1 : α) := isUnit_one.neg


theorem IsUnit.sub_iff [Ring α] {x y : α} : IsUnit (x - y) ↔ IsUnit (y - x) :=
  (IsUnit.neg_iff _).symm.trans <| neg_sub x y ▸ Iff.rfl


@[field_simps]
theorem divp_add_divp [CommRing α] (a b : α) (u₁ u₂ : αˣ) :
    a /ₚ u₁ + b /ₚ u₂ = (a * u₂ + u₁ * b) /ₚ (u₁ * u₂) := by
  /-
    α : Type u
    inst✝ : CommRing α
    a b : α
    u₁ u₂ : Units α
    ⊢ Eq (HAdd.hAdd (divp a u₁) (divp b u₂)) (divp (HAdd.hAdd (HMul.hMul a ↑u₂) (H …
  -/
  simp only [divp, add_mul, mul_inv_rev, val_mul]
  /-
    α : Type u
    inst✝ : CommRing α
    a b : α
    u₁ u₂ : Units α
    ⊢ Eq (HAdd.hAdd (HMul.hMul a ↑(Inv.inv u₁)) (HMul.hMul b ↑(Inv.inv u₂))) (HAdd …
  -/
  rw [mul_comm (↑u₁ * b), mul_comm b]
  rw [← mul_assoc, ← mul_assoc, mul_assoc a, mul_assoc (↑u₂⁻¹ : α), mul_inv, inv_mul, mul_one,
    mul_one]
  -- Porting note: `assoc_rw` not ported: `assoc_rw [mul_inv, mul_inv, mul_one, mul_one]`


@[field_simps]
theorem divp_sub_divp [CommRing α] (a b : α) (u₁ u₂ : αˣ) :
    a /ₚ u₁ - b /ₚ u₂ = (a * u₂ - u₁ * b) /ₚ (u₁ * u₂) := by
  /-
    α : Type u
    inst✝ : CommRing α
    a b : α
    u₁ u₂ : Units α
    ⊢ Eq (HSub.hSub (divp a u₁) (divp b u₂)) (divp (HSub.hSub (HMul.hMul a ↑u₂) (H …
  -/
  simp only [sub_eq_add_neg, neg_divp, divp_add_divp, mul_neg]
  /-
    🎉 no goals
  -/


theorem add_eq_mul_one_add_div [Semiring R] {a : Rˣ} {b : R} : ↑a + b = a * (1 + ↑a⁻¹ * b) := by
  /-
    R : Type x
    inst✝ : Semiring R
    a : Units R
    b : R
    ⊢ Eq (HAdd.hAdd (↑a) b) (HMul.hMul (↑a) (HAdd.hAdd 1 (HMul.hMul (↑(Inv.inv a)) …
  -/
  rw [mul_add, mul_one, ← mul_assoc, Units.mul_inv, one_mul]
  /-
    🎉 no goals
  -/


theorem isUnit_map (f : α →+* β) {a : α} : IsUnit a → IsUnit (f a) :=
  IsUnit.map f


