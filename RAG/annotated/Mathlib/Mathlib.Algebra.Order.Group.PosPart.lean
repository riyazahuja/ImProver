/-- The *positive part* of an element `a` in a lattice ordered group is `a ⊔ 1`, denoted `a⁺ᵐ`. -/
@[to_additive
"The *positive part* of an element `a` in a lattice ordered group is `a ⊔ 0`, denoted `a⁺`."]
instance instOneLePart : OneLePart α where
  oneLePart a := a ⊔ 1


/-- The *negative part* of an element `a` in a lattice ordered group is `a⁻¹ ⊔ 1`, denoted `a⁻ᵐ `.
-/
@[to_additive
"The *negative part* of an element `a` in a lattice ordered group is `(-a) ⊔ 0`, denoted `a⁻`."]
instance instLeOnePart : LeOnePart α where
  leOnePart a := a⁻¹ ⊔ 1


@[to_additive] lemma leOnePart_def (a : α) : a⁻ᵐ = a⁻¹ ⊔ 1 := rfl


@[to_additive] lemma oneLePart_def (a : α) : a⁺ᵐ = a ⊔ 1 := rfl


@[to_additive] lemma oneLePart_mono : Monotone (·⁺ᵐ : α → α) :=
  fun _a _b hab ↦ sup_le_sup_right hab _


@[to_additive (attr := simp high)] lemma oneLePart_one : (1 : α)⁺ᵐ = 1 := sup_idem _


                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : Lattice α
                                                                          inst✝ : Group α
                                                                          ⊢ Eq (LeOnePart.leOnePart 1) 1
                                                                        -/
@[to_additive (attr := simp)] lemma leOnePart_one : (1 : α)⁻ᵐ = 1 := by simp [leOnePart]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[to_additive posPart_nonneg] lemma one_le_oneLePart (a : α) : 1 ≤ a⁺ᵐ := le_sup_right


@[to_additive negPart_nonneg] lemma one_le_leOnePart (a : α) : 1 ≤ a⁻ᵐ := le_sup_right

-- TODO: `to_additive` guesses `nonposPart`

@[to_additive le_posPart] lemma le_oneLePart (a : α) : a ≤ a⁺ᵐ := le_sup_left


@[to_additive] lemma inv_le_leOnePart (a : α) : a⁻¹ ≤ a⁻ᵐ := le_sup_left


@[to_additive (attr := simp)] lemma oneLePart_eq_self : a⁺ᵐ = a ↔ 1 ≤ a := sup_eq_left

@[to_additive (attr := simp)] lemma oneLePart_eq_one : a⁺ᵐ = 1 ↔ a ≤ 1 := sup_eq_right


@[to_additive (attr := simp)] alias ⟨_, oneLePart_of_one_le⟩ := oneLePart_eq_self

@[to_additive (attr := simp)] alias ⟨_, oneLePart_of_le_one⟩ := oneLePart_eq_one


/-- See also `leOnePart_eq_inv`. -/
@[to_additive "See also `negPart_eq_neg`."]
lemma leOnePart_eq_inv' : a⁻ᵐ = a⁻¹ ↔ 1 ≤ a⁻¹ := sup_eq_left


/-- See also `leOnePart_eq_one`. -/
@[to_additive "See also `negPart_eq_zero`."]
lemma leOnePart_eq_one' : a⁻ᵐ = 1 ↔ a⁻¹ ≤ 1 := sup_eq_right


                                                              /-
                                                                α : Type u_1
                                                                inst✝¹ : Lattice α
                                                                inst✝ : Group α
                                                                a : α
                                                                ⊢ Iff (LE.le (OneLePart.oneLePart a) 1) (LE.le a 1)
                                                              -/
@[to_additive] lemma oneLePart_le_one : a⁺ᵐ ≤ 1 ↔ a ≤ 1 := by simp [oneLePart]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- See also `leOnePart_le_one`. -/
@[to_additive "See also `negPart_nonpos`."]
                                                  /-
                                                    α : Type u_1
                                                    inst✝¹ : Lattice α
                                                    inst✝ : Group α
                                                    a : α
                                                    ⊢ Iff (LE.le (LeOnePart.leOnePart a) 1) (LE.le (Inv.inv a) 1)
                                                  -/
lemma leOnePart_le_one' : a⁻ᵐ ≤ 1 ↔ a⁻¹ ≤ 1 := by simp [leOnePart]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                                /-
                                                                  α : Type u_1
                                                                  inst✝¹ : Lattice α
                                                                  inst✝ : Group α
                                                                  a : α
                                                                  ⊢ Iff (LE.le (LeOnePart.leOnePart a) 1) (LE.le (Inv.inv a) 1)
                                                                -/
@[to_additive] lemma leOnePart_le_one : a⁻ᵐ ≤ 1 ↔ a⁻¹ ≤ 1 := by simp [leOnePart]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive (attr := simp) posPart_pos] lemma one_lt_oneLePart (ha : 1 < a) : 1 < a⁺ᵐ := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : Group α
    a : α
    ha : LT.lt 1 a
    ⊢ LT.lt 1 (OneLePart.oneLePart a)
  -/
  rwa [oneLePart_eq_self.2 ha.le]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)] lemma oneLePart_inv (a : α) : a⁻¹⁺ᵐ = a⁻ᵐ := rfl


@[to_additive (attr := simp)] lemma leOnePart_inv (a : α) : a⁻¹⁻ᵐ = a⁺ᵐ := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : Group α
    a : α
    ⊢ Eq (LeOnePart.leOnePart (Inv.inv a)) (OneLePart.oneLePart a)
  -/
  simp [oneLePart, leOnePart]
  /-
    🎉 no goals
  -/


                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝² : Lattice α
                                                                                 inst✝¹ : Group α
                                                                                 a : α
                                                                                 inst✝ : MulLeftMono α
                                                                                 ⊢ Iff (Eq (LeOnePart.leOnePart a) (Inv.inv a)) (LE.le a 1)
                                                                               -/
@[to_additive (attr := simp)] lemma leOnePart_eq_inv : a⁻ᵐ = a⁻¹ ↔ a ≤ 1 := by simp [leOnePart]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive (attr := simp)]
                                               /-
                                                 α : Type u_1
                                                 inst✝² : Lattice α
                                                 inst✝¹ : Group α
                                                 a : α
                                                 inst✝ : MulLeftMono α
                                                 ⊢ Iff (Eq (LeOnePart.leOnePart a) 1) (LE.le 1 a)
                                               -/
lemma leOnePart_eq_one : a⁻ᵐ = 1 ↔ 1 ≤ a := by simp [leOnePart_eq_one']
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive (attr := simp)] alias ⟨_, leOnePart_of_le_one⟩ := leOnePart_eq_inv

@[to_additive (attr := simp)] alias ⟨_, leOnePart_of_one_le⟩ := leOnePart_eq_one


@[to_additive (attr := simp) negPart_pos] lemma one_lt_ltOnePart (ha : a < 1) : 1 < a⁻ᵐ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : Group α
    a : α
    inst✝ : MulLeftMono α
    ha : LT.lt a 1
    ⊢ LT.lt 1 (LeOnePart.leOnePart a)
  -/
  rwa [leOnePart_eq_inv.2 ha.le, one_lt_inv']
  /-
    🎉 no goals
  -/

-- Bourbaki A.VI.12 Prop 9 a)

@[to_additive (attr := simp)] lemma oneLePart_div_leOnePart (a : α) : a⁺ᵐ / a⁻ᵐ = a := by
  rw [div_eq_mul_inv, mul_inv_eq_iff_eq_mul, leOnePart_def, mul_sup, mul_one, mul_inv_cancel,
    sup_comm, oneLePart_def]


@[to_additive (attr := simp)] lemma leOnePart_div_oneLePart (a : α) : a⁻ᵐ / a⁺ᵐ = a⁻¹ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : Group α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HDiv.hDiv (LeOnePart.leOnePart a) (OneLePart.oneLePart a)) (Inv.inv a)
  -/
  rw [← inv_div, oneLePart_div_leOnePart]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma oneLePart_leOnePart_injective : Injective fun a : α ↦ (a⁺ᵐ, a⁻ᵐ) := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : Group α
    inst✝ : MulLeftMono α
    ⊢ Function.Injective fun a => { fst := OneLePart.oneLePart a, snd := LeOnePart …
  -/
  simp only [Injective, Prod.mk.injEq, and_imp]
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : Group α
    inst✝ : MulLeftMono α
    ⊢ ∀ ⦃a₁ a₂ : α⦄, Eq (OneLePart.oneLePart a₁) (OneLePart.oneLePart a₂) → Eq (Le …
  -/
  rintro a b hpos hneg
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : Group α
    inst✝ : MulLeftMono α
    a b : α
    hpos : Eq (OneLePart.oneLePart a) (OneLePart.oneLePart b)
    hneg : Eq (LeOnePart.leOnePart a) (LeOnePart.leOnePart b)
    ⊢ Eq a b
  -/
  rw [← oneLePart_div_leOnePart a, ← oneLePart_div_leOnePart b, hpos, hneg]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma oneLePart_leOnePart_inj : a⁺ᵐ = b⁺ᵐ ∧ a⁻ᵐ = b⁻ᵐ ↔ a = b :=
  Prod.mk.inj_iff.symm.trans oneLePart_leOnePart_injective.eq_iff


@[to_additive] lemma leOnePart_anti : Antitone (leOnePart : α → α) :=
  fun _a _b hab ↦ sup_le_sup_right (inv_le_inv_iff.2 hab) _


@[to_additive]
lemma leOnePart_eq_inv_inf_one (a : α) : a⁻ᵐ = (a ⊓ 1)⁻¹ := by
  /-
    α : Type u_1
    inst✝³ : Lattice α
    inst✝² : Group α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a : α
    ⊢ Eq (LeOnePart.leOnePart a) (Inv.inv (Min.min a 1))
  -/
  rw [leOnePart_def, ← inv_inj, inv_sup, inv_inv, inv_inv, inv_one]
  /-
    🎉 no goals
  -/

-- Bourbaki A.VI.12 Prop 9 d)

@[to_additive] lemma oneLePart_mul_leOnePart (a : α) : a⁺ᵐ * a⁻ᵐ = |a|ₘ := by
  rw [oneLePart_def, sup_mul, one_mul, leOnePart_def, mul_sup, mul_one, mul_inv_cancel, sup_assoc,
    ← sup_assoc a, sup_eq_right.2 le_sup_right]
  /-
    α : Type u_1
    inst✝³ : Lattice α
    inst✝² : Group α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a : α
    ⊢ Eq (Max.max (Max.max a (Inv.inv a)) 1) (mabs a)
  -/
  exact sup_eq_left.2 <| one_le_mabs a
  /-
    🎉 no goals
  -/


@[to_additive] lemma leOnePart_mul_oneLePart (a : α) : a⁻ᵐ * a⁺ᵐ = |a|ₘ := by
  rw [oneLePart_def, mul_sup, mul_one, leOnePart_def, sup_mul, one_mul, inv_mul_cancel, sup_assoc,
    ← @sup_assoc _ _ a, sup_eq_right.2 le_sup_right]
  /-
    α : Type u_1
    inst✝³ : Lattice α
    inst✝² : Group α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a : α
    ⊢ Eq (Max.max (Max.max a (Inv.inv a)) 1) (mabs a)
  -/
  exact sup_eq_left.2 <| one_le_mabs a
  /-
    🎉 no goals
  -/

-- Bourbaki A.VI.12 Prop 9 a)
-- a⁺ᵐ ⊓ a⁻ᵐ = 0 (`a⁺` and `a⁻` are co-prime, and, since they are positive, disjoint)

@[to_additive] lemma oneLePart_inf_leOnePart_eq_one (a : α) : a⁺ᵐ ⊓ a⁻ᵐ = 1 := by
  rw [← mul_left_inj a⁻ᵐ⁻¹, inf_mul, one_mul, mul_inv_cancel, ← div_eq_mul_inv,
    oneLePart_div_leOnePart, leOnePart_eq_inv_inf_one, inv_inv]


@[to_additive] lemma sup_eq_mul_oneLePart_div (a b : α) : a ⊔ b = b * (a / b)⁺ᵐ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a b : α
    ⊢ Eq (Max.max a b) (HMul.hMul b (OneLePart.oneLePart (HDiv.hDiv a b)))
  -/
  simp [oneLePart, mul_sup]
  /-
    🎉 no goals
  -/

-- Bourbaki A.VI.12 (with a and b swapped)

@[to_additive] lemma inf_eq_div_oneLePart_div (a b : α) : a ⊓ b = a / (a / b)⁺ᵐ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a b : α
    ⊢ Eq (Min.min a b) (HDiv.hDiv a (OneLePart.oneLePart (HDiv.hDiv a b)))
  -/
  simp [oneLePart, div_sup, inf_comm]
  /-
    🎉 no goals
  -/

-- Bourbaki A.VI.12 Prop 9 c)

@[to_additive] lemma le_iff_oneLePart_leOnePart (a b : α) : a ≤ b ↔ a⁺ᵐ ≤ b⁺ᵐ ∧ b⁻ᵐ ≤ a⁻ᵐ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a b : α
    ⊢ Iff (LE.le a b) (And (LE.le (OneLePart.oneLePart a) (OneLePart.oneLePart b)) …
  -/
  refine ⟨fun h ↦ ⟨oneLePart_mono h, leOnePart_anti h⟩, fun h ↦ ?_⟩
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a b : α
    h : And (LE.le (OneLePart.oneLePart a) (OneLePart.oneLePart b)) (LE.le (LeOneP …
    ⊢ LE.le a b
  -/
  rw [← oneLePart_div_leOnePart a, ← oneLePart_div_leOnePart b]
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a b : α
    h : And (LE.le (OneLePart.oneLePart a) (OneLePart.oneLePart b)) (LE.le (LeOneP …
    ⊢ LE.le (HDiv.hDiv (OneLePart.oneLePart a) (LeOnePart.leOnePart a)) (HDiv.hDiv …
  -/
  exact div_le_div'' h.1 h.2
  /-
    🎉 no goals
  -/


@[to_additive abs_add_eq_two_nsmul_posPart]
lemma mabs_mul_eq_oneLePart_sq (a : α) : |a|ₘ * a = a⁺ᵐ ^ 2 := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HMul.hMul (mabs a) a) (HPow.hPow (OneLePart.oneLePart a) 2)
  -/
  rw [sq, ← mul_mul_div_cancel a⁺ᵐ, oneLePart_mul_leOnePart, oneLePart_div_leOnePart]
  /-
    🎉 no goals
  -/


@[to_additive add_abs_eq_two_nsmul_posPart]
lemma mul_mabs_eq_oneLePart_sq (a : α) : a * |a|ₘ = a⁺ᵐ ^ 2 := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HMul.hMul a (mabs a)) (HPow.hPow (OneLePart.oneLePart a) 2)
  -/
  rw [mul_comm, mabs_mul_eq_oneLePart_sq]
  /-
    🎉 no goals
  -/


@[to_additive abs_sub_eq_two_nsmul_negPart]
lemma mabs_div_eq_leOnePart_sq (a : α) : |a|ₘ / a = a⁻ᵐ ^ 2 := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HDiv.hDiv (mabs a) a) (HPow.hPow (LeOnePart.leOnePart a) 2)
  -/
  rw [sq, ← mul_div_div_cancel, oneLePart_mul_leOnePart, oneLePart_div_leOnePart]
  /-
    🎉 no goals
  -/


@[to_additive sub_abs_eq_neg_two_nsmul_negPart]
lemma div_mabs_eq_inv_leOnePart_sq (a : α) : a / |a|ₘ = (a⁻ᵐ ^ 2)⁻¹ := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : CommGroup α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HDiv.hDiv a (mabs a)) (Inv.inv (HPow.hPow (LeOnePart.leOnePart a) 2))
  -/
  rw [← mabs_div_eq_leOnePart_sq, inv_div]
  /-
    🎉 no goals
  -/


@[to_additive] lemma oneLePart_eq_ite : a⁺ᵐ = if 1 ≤ a then a else 1 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Group α
    a : α
    ⊢ Eq (OneLePart.oneLePart a) (ite (LE.le 1 a) a 1)
  -/
  rw [oneLePart_def, ← maxDefault, ← sup_eq_maxDefault]; simp_rw [sup_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive (attr := simp) posPart_pos_iff] lemma one_lt_oneLePart_iff : 1 < a⁺ᵐ ↔ 1 < a :=
  lt_iff_lt_of_le_iff_le <| (one_le_oneLePart _).le_iff_eq.trans oneLePart_eq_one


@[to_additive posPart_eq_of_posPart_pos]
lemma oneLePart_of_one_lt_oneLePart (ha : 1 < a⁺ᵐ) : a⁺ᵐ = a := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Group α
    a : α
    ha : LT.lt 1 (OneLePart.oneLePart a)
    ⊢ Eq (OneLePart.oneLePart a) a
  -/
  rw [oneLePart_def, right_lt_sup, not_le] at ha; exact oneLePart_eq_self.2 ha.le
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive (attr := simp)] lemma oneLePart_lt : a⁺ᵐ < b ↔ a < b ∧ 1 < b := sup_lt_iff


@[to_additive] lemma leOnePart_eq_ite : a⁻ᵐ = if a ≤ 1 then a⁻¹ else 1 := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : Group α
    a : α
    inst✝ : MulLeftMono α
    ⊢ Eq (LeOnePart.leOnePart a) (ite (LE.le a 1) (Inv.inv a) 1)
  -/
  simp_rw [← one_le_inv']; rw [leOnePart_def, ← maxDefault, ← sup_eq_maxDefault]; simp_rw [sup_comm]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive (attr := simp) negPart_pos_iff] lemma one_lt_ltOnePart_iff : 1 < a⁻ᵐ ↔ a < 1 :=
  lt_iff_lt_of_le_iff_le <| (one_le_leOnePart _).le_iff_eq.trans leOnePart_eq_one


@[to_additive (attr := simp)] lemma leOnePart_lt : a⁻ᵐ < b ↔ b⁻¹ < a ∧ 1 < b :=
                         /-
                           α : Type u_1
                           inst✝³ : LinearOrder α
                           inst✝² : Group α
                           a b : α
                           inst✝¹ : MulLeftMono α
                           inst✝ : MulRightMono α
                           ⊢ Iff (And (LT.lt (Inv.inv a) b) (LT.lt 1 b)) (And (LT.lt (Inv.inv b) a) (LT.l …
                         -/
  sup_lt_iff.trans <| by rw [inv_lt']
                         /-
                           🎉 no goals
                         -/


@[to_additive (attr := simp)] lemma oneLePart_apply (f : ∀ i, α i) (i : ι) : f⁺ᵐ i = (f i)⁺ᵐ := rfl

@[to_additive (attr := simp)] lemma leOnePart_apply (f : ∀ i, α i) (i : ι) : f⁻ᵐ i = (f i)⁻ᵐ := rfl


@[to_additive] lemma oneLePart_def (f : ∀ i, α i) : f⁺ᵐ = fun i ↦ (f i)⁺ᵐ := rfl

@[to_additive] lemma leOnePart_def (f : ∀ i, α i) : f⁻ᵐ = fun i ↦ (f i)⁻ᵐ := rfl


