@[to_additive]
theorem unique_one {α : Type*} [Unique α] [One α] : default = (1 : α) :=
  Unique.default_eq 1


@[to_additive (attr := simp)]
theorem mul_inv_cancel_right (a : α) (b : αˣ) : a * b * ↑b⁻¹ = a := by
  /-
    α : Type u
    inst✝ : Monoid α
    a : α
    b : Units α
    ⊢ Eq (HMul.hMul (HMul.hMul a ↑b) ↑(Inv.inv b)) a
  -/
  rw [mul_assoc, mul_inv, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inv_mul_cancel_right (a : α) (b : αˣ) : a * ↑b⁻¹ * b = a := by
  /-
    α : Type u
    inst✝ : Monoid α
    a : α
    b : Units α
    ⊢ Eq (HMul.hMul (HMul.hMul a ↑(Inv.inv b)) ↑b) a
  -/
  rw [mul_assoc, inv_mul, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_right_inj (a : αˣ) {b c : α} : (a : α) * b = a * c ↔ b = c :=
               /-
                 α : Type u
                 inst✝ : Monoid α
                 a : Units α
                 b c : α
                 h : Eq (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
                 ⊢ Eq b c
               -/
  ⟨fun h => by simpa only [inv_mul_cancel_left] using congr_arg (fun x : α => ↑(a⁻¹ : αˣ) * x) h,
               /-
                 🎉 no goals
               -/
    congr_arg _⟩


@[to_additive (attr := simp)]
theorem mul_left_inj (a : αˣ) {b c : α} : b * a = c * a ↔ b = c :=
               /-
                 α : Type u
                 inst✝ : Monoid α
                 a : Units α
                 b c : α
                 h : Eq (HMul.hMul b ↑a) (HMul.hMul c ↑a)
                 ⊢ Eq b c
               -/
  ⟨fun h => by simpa only [mul_inv_cancel_right] using congr_arg (fun x : α => x * ↑(a⁻¹ : αˣ)) h,
               /-
                 🎉 no goals
               -/
    congr_arg (· * a.val)⟩


@[to_additive]
theorem eq_mul_inv_iff_mul_eq {a b : α} : a = b * ↑c⁻¹ ↔ a * c = b :=
               /-
                 α : Type u
                 inst✝ : Monoid α
                 c : Units α
                 a b : α
                 h : Eq a (HMul.hMul b ↑(Inv.inv c))
                 ⊢ Eq (HMul.hMul a ↑c) b
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [h, inv_mul_cancel_right], fun h => by rw [← h, mul_inv_cancel_right]⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
theorem eq_inv_mul_iff_mul_eq {a c : α} : a = ↑b⁻¹ * c ↔ ↑b * a = c :=
               /-
                 α : Type u
                 inst✝ : Monoid α
                 b : Units α
                 a c : α
                 h : Eq a (HMul.hMul (↑(Inv.inv b)) c)
                 ⊢ Eq (HMul.hMul (↑b) a) c
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [h, mul_inv_cancel_left], fun h => by rw [← h, inv_mul_cancel_left]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
theorem mul_inv_eq_iff_eq_mul {a c : α} : a * ↑b⁻¹ = c ↔ a = c * b :=
               /-
                 α : Type u
                 inst✝ : Monoid α
                 b : Units α
                 a c : α
                 h : Eq (HMul.hMul a ↑(Inv.inv b)) c
                 ⊢ Eq a (HMul.hMul c ↑b)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [← h, inv_mul_cancel_right], fun h => by rw [h, mul_inv_cancel_right]⟩
                                                           /-
                                                             🎉 no goals
                                                           -/

-- Porting note: have to explicitly type annotate the 1

@[to_additive]
protected theorem inv_eq_of_mul_eq_one_left {a : α} (h : a * u = 1) : ↑u⁻¹ = a :=
  calc
                                /-
                                  α : Type u
                                  inst✝ : Monoid α
                                  u : Units α
                                  a : α
                                  h : Eq (HMul.hMul a ↑u) 1
                                  ⊢ Eq (↑(Inv.inv u)) (HMul.hMul 1 ↑(Inv.inv u))
                                -/
    ↑u⁻¹ = (1 : α) * ↑u⁻¹ := by rw [one_mul]
                                /-
                                  🎉 no goals
                                -/
                /-
                  α : Type u
                  inst✝ : Monoid α
                  u : Units α
                  a : α
                  h : Eq (HMul.hMul a ↑u) 1
                  ⊢ Eq (HMul.hMul 1 ↑(Inv.inv u)) a
                -/
    _ = a := by rw [← h, mul_inv_cancel_right]
                /-
                  🎉 no goals
                -/


-- Porting note: have to explicitly type annotate the 1

@[to_additive]
protected theorem inv_eq_of_mul_eq_one_right {a : α} (h : ↑u * a = 1) : ↑u⁻¹ = a :=
  calc
                                /-
                                  α : Type u
                                  inst✝ : Monoid α
                                  u : Units α
                                  a : α
                                  h : Eq (HMul.hMul (↑u) a) 1
                                  ⊢ Eq (↑(Inv.inv u)) (HMul.hMul (↑(Inv.inv u)) 1)
                                -/
    ↑u⁻¹ = ↑u⁻¹ * (1 : α) := by rw [mul_one]
                                /-
                                  🎉 no goals
                                -/
                /-
                  α : Type u
                  inst✝ : Monoid α
                  u : Units α
                  a : α
                  h : Eq (HMul.hMul (↑u) a) 1
                  ⊢ Eq (HMul.hMul (↑(Inv.inv u)) 1) a
                -/
    _ = a := by rw [← h, inv_mul_cancel_left]
                /-
                  🎉 no goals
                -/



@[to_additive]
protected theorem eq_inv_of_mul_eq_one_left {a : α} (h : ↑u * a = 1) : a = ↑u⁻¹ :=
  (Units.inv_eq_of_mul_eq_one_right h).symm


@[to_additive]
protected theorem eq_inv_of_mul_eq_one_right {a : α} (h : a * u = 1) : a = ↑u⁻¹ :=
  (Units.inv_eq_of_mul_eq_one_left h).symm


@[to_additive (attr := simp)]
theorem mul_inv_eq_one {a : α} : a * ↑u⁻¹ = 1 ↔ a = u :=
  ⟨inv_inv u ▸ Units.eq_inv_of_mul_eq_one_right, fun h => mul_inv_of_eq h.symm⟩


@[to_additive (attr := simp)]
theorem inv_mul_eq_one {a : α} : ↑u⁻¹ * a = 1 ↔ ↑u = a :=
  ⟨inv_inv u ▸ Units.inv_eq_of_mul_eq_one_right, inv_mul_of_eq⟩


@[to_additive]
                                                                   /-
                                                                     α : Type u
                                                                     inst✝ : Monoid α
                                                                     u : Units α
                                                                     a : α
                                                                     ⊢ Iff (Eq (HMul.hMul a ↑u) 1) (Eq a ↑(Inv.inv u))
                                                                   -/
theorem mul_eq_one_iff_eq_inv {a : α} : a * u = 1 ↔ a = ↑u⁻¹ := by rw [← mul_inv_eq_one, inv_inv]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive]
                                                                    /-
                                                                      α : Type u
                                                                      inst✝ : Monoid α
                                                                      u : Units α
                                                                      a : α
                                                                      ⊢ Iff (Eq (HMul.hMul (↑u) a) 1) (Eq (↑(Inv.inv u)) a)
                                                                    -/
theorem mul_eq_one_iff_inv_eq {a : α} : ↑u * a = 1 ↔ ↑u⁻¹ = a := by rw [← inv_mul_eq_one, inv_inv]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive]
theorem inv_unique {u₁ u₂ : αˣ} (h : (↑u₁ : α) = ↑u₂) : (↑u₁⁻¹ : α) = ↑u₂⁻¹ :=
                                         /-
                                           α : Type u
                                           inst✝ : Monoid α
                                           u₁ u₂ : Units α
                                           h : Eq ↑u₁ ↑u₂
                                           ⊢ Eq (HMul.hMul ↑u₁ ↑(Inv.inv u₂)) 1
                                         -/
  Units.inv_eq_of_mul_eq_one_right <| by rw [h, u₂.mul_inv]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem divp_left_inj (u : αˣ) {a b : α} : a /ₚ u = b /ₚ u ↔ a = b :=
  Units.mul_left_inj _

/- Porting note: to match the mathlib3 behavior, this needs to have higher simp
priority than eq_divp_iff_mul_eq. -/

@[field_simps 1010]
theorem divp_eq_iff_mul_eq {x : α} {u : αˣ} {y : α} : x /ₚ u = y ↔ y * u = x :=
                                  /-
                                    α : Type u
                                    inst✝ : Monoid α
                                    x : α
                                    u : Units α
                                    y : α
                                    ⊢ Iff (Eq (HMul.hMul (divp x u) ↑u) (HMul.hMul y ↑u)) (Eq (HMul.hMul y ↑u) x)
                                  -/
  u.mul_left_inj.symm.trans <| by rw [divp_mul_cancel]; exact ⟨Eq.symm, Eq.symm⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[field_simps]
theorem eq_divp_iff_mul_eq {x : α} {u : αˣ} {y : α} : x = y /ₚ u ↔ x * u = y := by
  /-
    α : Type u
    inst✝ : Monoid α
    x : α
    u : Units α
    y : α
    ⊢ Iff (Eq x (divp y u)) (Eq (HMul.hMul x ↑u) y)
  -/
  rw [eq_comm, divp_eq_iff_mul_eq]
  /-
    🎉 no goals
  -/


theorem divp_eq_one_iff_eq {a : α} {u : αˣ} : a /ₚ u = 1 ↔ a = u :=
                                          /-
                                            α : Type u
                                            inst✝ : Monoid α
                                            a : α
                                            u : Units α
                                            ⊢ Iff (Eq (HMul.hMul (divp a u) ↑u) (HMul.hMul 1 ↑u)) (Eq a ↑u)
                                          -/
  (Units.mul_left_inj u).symm.trans <| by rw [divp_mul_cancel, one_mul]
                                          /-
                                            🎉 no goals
                                          -/


/-- Used for `field_simp` to deal with inverses of units. This form of the lemma
is essential since `field_simp` likes to use `inv_eq_one_div` to rewrite
`↑u⁻¹ = ↑(1 / u)`.
-/
@[field_simps]
theorem inv_eq_one_divp' (u : αˣ) : ((1 / u : αˣ) : α) = 1 /ₚ u := by
  /-
    α : Type u
    inst✝ : Monoid α
    u : Units α
    ⊢ Eq (↑(HDiv.hDiv 1 u)) (divp 1 u)
  -/
  rw [one_div, one_divp]
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem eq_one_of_mul_right (h : a * b = 1) : a = 1 :=
  congr_arg Units.inv <| Subsingleton.elim (Units.mk _ _ (by
    /-
      α : Type u
      inst✝¹ : LeftCancelMonoid α
      inst✝ : Subsingleton (Units α)
      a b : α
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul b a) 1
    -/
    rw [← mul_left_cancel_iff (a := a), ← mul_assoc, h, one_mul, mul_one]) h) 1
    /-
      🎉 no goals
    -/


@[to_additive]
protected theorem eq_one_of_mul_left (h : a * b = 1) : b = 1 := by
  /-
    α : Type u
    inst✝¹ : LeftCancelMonoid α
    inst✝ : Subsingleton (Units α)
    a b : α
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq b 1
  -/
  rwa [LeftCancelMonoid.eq_one_of_mul_right h, one_mul] at h
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem mul_eq_one : a * b = 1 ↔ a = 1 ∧ b = 1 :=
  ⟨fun h => ⟨LeftCancelMonoid.eq_one_of_mul_right h, LeftCancelMonoid.eq_one_of_mul_left h⟩, by
    /-
      α : Type u
      inst✝¹ : LeftCancelMonoid α
      inst✝ : Subsingleton (Units α)
      a b : α
      ⊢ And (Eq a 1) (Eq b 1) → Eq (HMul.hMul a b) 1
    -/
    rintro ⟨rfl, rfl⟩
    /-
      case intro
      α : Type u
      inst✝¹ : LeftCancelMonoid α
      inst✝ : Subsingleton (Units α)
      ⊢ Eq (HMul.hMul 1 1) 1
    -/
    exact mul_one _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
                                                               /-
                                                                 α : Type u
                                                                 inst✝¹ : LeftCancelMonoid α
                                                                 inst✝ : Subsingleton (Units α)
                                                                 a b : α
                                                                 ⊢ Iff (Ne (HMul.hMul a b) 1) (Or (Ne a 1) (Ne b 1))
                                                               -/
protected theorem mul_ne_one : a * b ≠ 1 ↔ a ≠ 1 ∨ b ≠ 1 := by rw [not_iff_comm]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive]
protected theorem eq_one_of_mul_right (h : a * b = 1) : a = 1 :=
  congr_arg Units.inv <| Subsingleton.elim (Units.mk _ _ (by
    /-
      α : Type u
      inst✝¹ : RightCancelMonoid α
      inst✝ : Subsingleton (Units α)
      a b : α
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul b a) 1
    -/
    rw [← mul_right_cancel_iff (a := b), mul_assoc, h, one_mul, mul_one]) h) 1
    /-
      🎉 no goals
    -/


@[to_additive]
protected theorem eq_one_of_mul_left (h : a * b = 1) : b = 1 := by
  /-
    α : Type u
    inst✝¹ : RightCancelMonoid α
    inst✝ : Subsingleton (Units α)
    a b : α
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq b 1
  -/
  rwa [RightCancelMonoid.eq_one_of_mul_right h, one_mul] at h
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem mul_eq_one : a * b = 1 ↔ a = 1 ∧ b = 1 :=
  ⟨fun h => ⟨RightCancelMonoid.eq_one_of_mul_right h, RightCancelMonoid.eq_one_of_mul_left h⟩, by
    /-
      α : Type u
      inst✝¹ : RightCancelMonoid α
      inst✝ : Subsingleton (Units α)
      a b : α
      ⊢ And (Eq a 1) (Eq b 1) → Eq (HMul.hMul a b) 1
    -/
    rintro ⟨rfl, rfl⟩
    /-
      case intro
      α : Type u
      inst✝¹ : RightCancelMonoid α
      inst✝ : Subsingleton (Units α)
      ⊢ Eq (HMul.hMul 1 1) 1
    -/
    exact mul_one _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
                                                               /-
                                                                 α : Type u
                                                                 inst✝¹ : RightCancelMonoid α
                                                                 inst✝ : Subsingleton (Units α)
                                                                 a b : α
                                                                 ⊢ Iff (Ne (HMul.hMul a b) 1) (Or (Ne a 1) (Ne b 1))
                                                               -/
protected theorem mul_ne_one : a * b ≠ 1 ↔ a ≠ 1 ∨ b ≠ 1 := by rw [not_iff_comm]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive]
theorem eq_one_of_mul_right' (h : a * b = 1) : a = 1 := LeftCancelMonoid.eq_one_of_mul_right h


@[to_additive]
theorem eq_one_of_mul_left' (h : a * b = 1) : b = 1 := LeftCancelMonoid.eq_one_of_mul_left h


@[to_additive]
theorem mul_eq_one' : a * b = 1 ↔ a = 1 ∧ b = 1 := LeftCancelMonoid.mul_eq_one


@[to_additive]
theorem mul_ne_one' : a * b ≠ 1 ↔ a ≠ 1 ∨ b ≠ 1 := LeftCancelMonoid.mul_ne_one


@[field_simps]
theorem divp_mul_eq_mul_divp (x y : α) (u : αˣ) : x /ₚ u * y = x * y /ₚ u := by
  /-
    α : Type u
    inst✝ : CommMonoid α
    x y : α
    u : Units α
    ⊢ Eq (HMul.hMul (divp x u) y) (divp (HMul.hMul x y) u)
  -/
  rw [divp, divp, mul_right_comm]
  /-
    🎉 no goals
  -/

-- Theoretically redundant as `field_simp` lemma.

@[field_simps]
theorem divp_eq_divp_iff {x y : α} {ux uy : αˣ} : x /ₚ ux = y /ₚ uy ↔ x * uy = y * ux := by
  /-
    α : Type u
    inst✝ : CommMonoid α
    x y : α
    ux uy : Units α
    ⊢ Iff (Eq (divp x ux) (divp y uy)) (Eq (HMul.hMul x ↑uy) (HMul.hMul y ↑ux))
  -/
  rw [divp_eq_iff_mul_eq, divp_mul_eq_mul_divp, divp_eq_iff_mul_eq]
  /-
    🎉 no goals
  -/

-- Theoretically redundant as `field_simp` lemma.

@[field_simps]
theorem divp_mul_divp (x y : α) (ux uy : αˣ) : x /ₚ ux * (y /ₚ uy) = x * y /ₚ (ux * uy) := by
  /-
    α : Type u
    inst✝ : CommMonoid α
    x y : α
    ux uy : Units α
    ⊢ Eq (HMul.hMul (divp x ux) (divp y uy)) (divp (HMul.hMul x y) (HMul.hMul ux u …
  -/
  rw [divp_mul_eq_mul_divp, divp_assoc', divp_divp_eq_divp_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_one_of_mul_right (h : a * b = 1) : a = 1 :=
                                                             /-
                                                               α : Type u
                                                               inst✝¹ : CommMonoid α
                                                               inst✝ : Subsingleton (Units α)
                                                               a b : α
                                                               h : Eq (HMul.hMul a b) 1
                                                               ⊢ Eq (HMul.hMul b a) 1
                                                             -/
  congr_arg Units.inv <| Subsingleton.elim (Units.mk _ _ (by rwa [mul_comm]) h) 1
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
theorem eq_one_of_mul_left (h : a * b = 1) : b = 1 :=
                                                                 /-
                                                                   α : Type u
                                                                   inst✝¹ : CommMonoid α
                                                                   inst✝ : Subsingleton (Units α)
                                                                   a b : α
                                                                   h : Eq (HMul.hMul a b) 1
                                                                   ⊢ Eq (HMul.hMul b a) 1
                                                                 -/
  congr_arg Units.inv <| Subsingleton.elim (Units.mk _ _ h <| by rwa [mul_comm]) 1
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive (attr := simp)]
theorem mul_eq_one : a * b = 1 ↔ a = 1 ∧ b = 1 :=
  ⟨fun h => ⟨eq_one_of_mul_right h, eq_one_of_mul_left h⟩, by
    /-
      α : Type u
      inst✝¹ : CommMonoid α
      inst✝ : Subsingleton (Units α)
      a b : α
      ⊢ And (Eq a 1) (Eq b 1) → Eq (HMul.hMul a b) 1
    -/
    rintro ⟨rfl, rfl⟩
    /-
      case intro
      α : Type u
      inst✝¹ : CommMonoid α
      inst✝ : Subsingleton (Units α)
      ⊢ Eq (HMul.hMul 1 1) 1
    -/
    exact mul_one _⟩
    /-
      🎉 no goals
    -/


                                                                    /-
                                                                      α : Type u
                                                                      inst✝¹ : CommMonoid α
                                                                      inst✝ : Subsingleton (Units α)
                                                                      a b : α
                                                                      ⊢ Iff (Ne (HMul.hMul a b) 1) (Or (Ne a 1) (Ne b 1))
                                                                    -/
@[to_additive] theorem mul_ne_one : a * b ≠ 1 ↔ a ≠ 1 ∨ b ≠ 1 := by rw [not_iff_comm]; simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[to_additive (attr := nontriviality)]
theorem isUnit_of_subsingleton [Monoid M] [Subsingleton M] (a : M) : IsUnit a :=
             /-
               M : Type u_1
               inst✝¹ : Monoid M
               inst✝ : Subsingleton M
               a : M
               ⊢ Eq (HMul.hMul a a) 1
             -/
             /-
               🎉 no goals
             -/
  ⟨⟨a, a, by subsingleton, by subsingleton⟩, rfl⟩
                              /-
                                🎉 no goals
                              -/


@[to_additive]
instance [Monoid M] : CanLift M Mˣ Units.val IsUnit :=
  { prf := fun _ ↦ id }


/-- A subsingleton `Monoid` has a unique unit. -/
@[to_additive "A subsingleton `AddMonoid` has a unique additive unit."]
instance [Monoid M] [Subsingleton M] : Unique Mˣ where
                                    /-
                                      α : Type u
                                      M : Type u_1
                                      inst✝¹ : Monoid M
                                      inst✝ : Subsingleton M
                                      x✝ : Units M
                                      ⊢ Eq (↑x✝) 1
                                    -/
  uniq _ := Units.val_eq_one.mp (by subsingleton)
                                    /-
                                      🎉 no goals
                                    -/


                                                              /-
                                                                M : Type u_1
                                                                inst✝¹ : Monoid M
                                                                inst✝ : Subsingleton (Units M)
                                                                u : Units M
                                                                ⊢ Eq u 1
                                                              -/
theorem units_eq_one [Subsingleton Mˣ] (u : Mˣ) : u = 1 := by subsingleton
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive]
theorem mul_left_inj (h : IsUnit a) : b * a = c * a ↔ b = c :=
  let ⟨u, hu⟩ := h
  hu ▸ u.mul_left_inj


@[to_additive]
theorem mul_right_inj (h : IsUnit a) : a * b = a * c ↔ b = c :=
  let ⟨u, hu⟩ := h
  hu ▸ u.mul_right_inj


@[to_additive]
protected theorem mul_left_cancel (h : IsUnit a) : a * b = a * c → b = c :=
  h.mul_right_inj.1


@[to_additive]
protected theorem mul_right_cancel (h : IsUnit b) : a * b = c * b → a = c :=
  h.mul_left_inj.1


@[to_additive]
protected theorem mul_right_injective (h : IsUnit a) : Injective (a * ·) :=
  fun _ _ => h.mul_left_cancel


@[to_additive]
protected theorem mul_left_injective (h : IsUnit b) : Injective (· * b) :=
  fun _ _ => h.mul_right_cancel


@[to_additive]
theorem isUnit_iff_mulLeft_bijective {a : M} :
    IsUnit a ↔ Function.Bijective (a * ·) :=
                                                             /-
                                                               M : Type u_1
                                                               inst✝ : Monoid M
                                                               a : M
                                                               h : IsUnit a
                                                               y : M
                                                               ⊢ Eq ((fun x => HMul.hMul a x) (HMul.hMul (↑(Inv.inv h.unit)) y)) y
                                                             -/
  ⟨fun h ↦ ⟨h.mul_right_injective, fun y ↦ ⟨h.unit⁻¹ * y, by simp [← mul_assoc]⟩⟩, fun h ↦
                                                             /-
                                                               🎉 no goals
                                                             -/
    ⟨⟨a, _, (h.2 1).choose_spec, h.1
          /-
            M : Type u_1
            inst✝ : Monoid M
            a : M
            h : Function.Bijective fun x => HMul.hMul a x
            ⊢ Eq ((fun x => HMul.hMul a x) (HMul.hMul ⋯.choose a)) ((fun x => HMul.hMul a  …
          -/
      (by simpa [mul_assoc] using congr_arg (· * a) (h.2 1).choose_spec)⟩, rfl⟩⟩
          /-
            🎉 no goals
          -/


@[to_additive]
theorem isUnit_iff_mulRight_bijective {a : M} :
    IsUnit a ↔ Function.Bijective (· * a) :=
                                                            /-
                                                              M : Type u_1
                                                              inst✝ : Monoid M
                                                              a : M
                                                              h : IsUnit a
                                                              y : M
                                                              ⊢ Eq ((fun x => HMul.hMul x a) (HMul.hMul y ↑(Inv.inv h.unit))) y
                                                            -/
  ⟨fun h ↦ ⟨h.mul_left_injective, fun y ↦ ⟨y * h.unit⁻¹, by simp [mul_assoc]⟩⟩,
                                                            /-
                                                              🎉 no goals
                                                            -/
                            /-
                              M : Type u_1
                              inst✝ : Monoid M
                              a : M
                              h : Function.Bijective fun x => HMul.hMul x a
                              ⊢ Eq ((fun x => HMul.hMul x a) (HMul.hMul a ⋯.choose)) ((fun x => HMul.hMul x  …
                            -/
    fun h ↦ ⟨⟨a, _, h.1 (by simpa [mul_assoc] using congr_arg (a * ·) (h.2 1).choose_spec),
                            /-
                              🎉 no goals
                            -/
      (h.2 1).choose_spec⟩, rfl⟩⟩


@[to_additive (attr := simp)]
protected lemma mul_inv_cancel_right (h : IsUnit b) (a : α) : a * b * b⁻¹ = a :=
  h.unit'.mul_inv_cancel_right _


@[to_additive (attr := simp)]
protected lemma inv_mul_cancel_right (h : IsUnit b) (a : α) : a * b⁻¹ * b = a :=
  h.unit'.inv_mul_cancel_right _


@[to_additive]
protected lemma eq_mul_inv_iff_mul_eq (h : IsUnit c) : a = b * c⁻¹ ↔ a * c = b :=
  h.unit'.eq_mul_inv_iff_mul_eq


@[to_additive]
protected lemma eq_inv_mul_iff_mul_eq (h : IsUnit b) : a = b⁻¹ * c ↔ b * a = c :=
  h.unit'.eq_inv_mul_iff_mul_eq


@[to_additive]
protected lemma inv_mul_eq_iff_eq_mul (h : IsUnit a) : a⁻¹ * b = c ↔ b = a * c :=
  h.unit'.inv_mul_eq_iff_eq_mul


@[to_additive]
protected lemma mul_inv_eq_iff_eq_mul (h : IsUnit b) : a * b⁻¹ = c ↔ a = c * b :=
  h.unit'.mul_inv_eq_iff_eq_mul


@[to_additive]
protected lemma mul_inv_eq_one (h : IsUnit b) : a * b⁻¹ = 1 ↔ a = b :=
  @Units.mul_inv_eq_one _ _ h.unit' _


@[to_additive]
protected lemma inv_mul_eq_one (h : IsUnit a) : a⁻¹ * b = 1 ↔ a = b :=
  @Units.inv_mul_eq_one _ _ h.unit' _


@[to_additive]
protected lemma mul_eq_one_iff_eq_inv (h : IsUnit b) : a * b = 1 ↔ a = b⁻¹ :=
  @Units.mul_eq_one_iff_eq_inv _ _ h.unit' _


@[to_additive]
protected lemma mul_eq_one_iff_inv_eq (h : IsUnit a) : a * b = 1 ↔ a⁻¹ = b :=
  @Units.mul_eq_one_iff_inv_eq _ _ h.unit' _


@[to_additive (attr := simp)]
protected lemma div_mul_cancel (h : IsUnit b) (a : α) : a / b * b = a := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    b : α
    h : IsUnit b
    a : α
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) b) a
  -/
  rw [div_eq_mul_inv, h.inv_mul_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected lemma mul_div_cancel_right (h : IsUnit b) (a : α) : a * b / b = a := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    b : α
    h : IsUnit b
    a : α
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) b) a
  -/
  rw [div_eq_mul_inv, h.mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                          /-
                                                                            α : Type u
                                                                            inst✝ : DivisionMonoid α
                                                                            a : α
                                                                            h : IsUnit a
                                                                            ⊢ Eq (HMul.hMul a (HDiv.hDiv 1 a)) 1
                                                                          -/
protected lemma mul_one_div_cancel (h : IsUnit a) : a * (1 / a) = 1 := by simp [h]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive]
                                                                        /-
                                                                          α : Type u
                                                                          inst✝ : DivisionMonoid α
                                                                          a : α
                                                                          h : IsUnit a
                                                                          ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) a) 1
                                                                        -/
protected lemma one_div_mul_cancel (h : IsUnit a) : 1 / a * a = 1 := by simp [h]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[to_additive]
protected lemma div_left_inj (h : IsUnit c) : a / c = b / c ↔ a = b := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    a b c : α
    h : IsUnit c
    ⊢ Iff (Eq (HDiv.hDiv a c) (HDiv.hDiv b c)) (Eq a b)
  -/
  simp only [div_eq_mul_inv]
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    a b c : α
    h : IsUnit c
    ⊢ Iff (Eq (HMul.hMul a (Inv.inv c)) (HMul.hMul b (Inv.inv c))) (Eq a b)
  -/
  exact Units.mul_left_inj h.inv.unit'
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma div_eq_iff (h : IsUnit b) : a / b = c ↔ a = c * b := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    a b c : α
    h : IsUnit b
    ⊢ Iff (Eq (HDiv.hDiv a b) c) (Eq a (HMul.hMul c b))
  -/
  rw [div_eq_mul_inv, h.mul_inv_eq_iff_eq_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma eq_div_iff (h : IsUnit c) : a = b / c ↔ a * c = b := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    a b c : α
    h : IsUnit c
    ⊢ Iff (Eq a (HDiv.hDiv b c)) (Eq (HMul.hMul a c) b)
  -/
  rw [div_eq_mul_inv, h.eq_mul_inv_iff_mul_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma div_eq_of_eq_mul (h : IsUnit b) : a = c * b → a / b = c :=
  h.div_eq_iff.2


@[to_additive]
protected lemma eq_div_of_mul_eq (h : IsUnit c) : a * c = b → a = b / c :=
  h.eq_div_iff.2


@[to_additive]
protected lemma div_eq_one_iff_eq (h : IsUnit b) : a / b = 1 ↔ a = b :=
  ⟨eq_of_div_eq_one, fun hab => hab.symm ▸ h.div_self⟩


@[to_additive]
protected lemma div_mul_left (h : IsUnit b) : b / (a * b) = 1 / a := by
  /-
    α : Type u
    inst✝ : DivisionMonoid α
    a b : α
    h : IsUnit b
    ⊢ Eq (HDiv.hDiv b (HMul.hMul a b)) (HDiv.hDiv 1 a)
  -/
  rw [h.div_mul_cancel_right, one_div]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                               /-
                                                                                 α : Type u
                                                                                 inst✝ : DivisionMonoid α
                                                                                 b a : α
                                                                                 h : IsUnit b
                                                                                 ⊢ Eq (HMul.hMul (HMul.hMul a b) (HDiv.hDiv 1 b)) a
                                                                               -/
protected lemma mul_mul_div (a : α) (h : IsUnit b) : a * b * (1 / b) = a := by simp [h]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive]
protected lemma div_mul_right (h : IsUnit a) (b : α) : a / (a * b) = 1 / b := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    a : α
    h : IsUnit a
    b : α
    ⊢ Eq (HDiv.hDiv a (HMul.hMul a b)) (HDiv.hDiv 1 b)
  -/
  rw [mul_comm, h.div_mul_left]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma mul_div_cancel_left (h : IsUnit a) (b : α) : a * b / a = b := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    a : α
    h : IsUnit a
    b : α
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) a) b
  -/
  rw [mul_comm, h.mul_div_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma mul_div_cancel (h : IsUnit a) (b : α) : a * (b / a) = b := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    a : α
    h : IsUnit a
    b : α
    ⊢ Eq (HMul.hMul a (HDiv.hDiv b a)) b
  -/
  rw [mul_comm, h.div_mul_cancel]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma mul_eq_mul_of_div_eq_div (hb : IsUnit b) (hd : IsUnit d)
    (a c : α) (h : a / b = c / d) : a * d = c * b := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    b d : α
    hb : IsUnit b
    hd : IsUnit d
    a c : α
    h : Eq (HDiv.hDiv a b) (HDiv.hDiv c d)
    ⊢ Eq (HMul.hMul a d) (HMul.hMul c b)
  -/
  rw [← mul_one a, ← hb.div_self, ← mul_comm_div, h, div_mul_eq_mul_div, hd.div_mul_cancel]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma div_eq_div_iff (hb : IsUnit b) (hd : IsUnit d) :
    a / b = c / d ↔ a * d = c * b := by
  rw [← (hb.mul hd).mul_left_inj, ← mul_assoc, hb.div_mul_cancel, ← mul_assoc, mul_right_comm,
    hd.div_mul_cancel]


@[to_additive]
protected lemma div_div_cancel (h : IsUnit a) : a / (a / b) = b := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    a b : α
    h : IsUnit a
    ⊢ Eq (HDiv.hDiv a (HDiv.hDiv a b)) b
  -/
  rw [div_div_eq_mul_div, h.mul_div_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive]
protected lemma div_div_cancel_left (h : IsUnit a) : a / b / a = b⁻¹ := by
  /-
    α : Type u
    inst✝ : DivisionCommMonoid α
    a b : α
    h : IsUnit a
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) a) (Inv.inv b)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, mul_right_comm, h.mul_inv_cancel, one_mul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-20")] alias IsUnit.mul_div_cancel' := IsUnit.mul_div_cancel

@[deprecated (since := "2024-03-20")] alias IsAddUnit.add_sub_cancel' := IsAddUnit.add_sub_cancel

