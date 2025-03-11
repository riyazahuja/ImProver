/-- An element `a` of a type `α` with multiplication satisfies `IsSquare a` if `a = r * r`,
for some `r : α`. -/
@[to_additive "An element `a` of a type `α` with addition satisfies `Even a` if `a = r + r`,
for some `r : α`."]
def IsSquare (a : α) : Prop := ∃ r, a = r * r


@[to_additive (attr := simp)] lemma IsSquare.mul_self (m : α) : IsSquare (m * m) := ⟨m, rfl⟩


@[deprecated (since := "2024-08-27")] alias isSquare_mul_self := IsSquare.mul_self

@[deprecated (since := "2024-08-27")] alias even_add_self := Even.add_self


@[to_additive]
lemma isSquare_op_iff {a : α} : IsSquare (op a) ↔ IsSquare a :=
  ⟨fun ⟨c, hc⟩ ↦ ⟨unop c, congr_arg unop hc⟩, fun ⟨c, hc⟩ ↦ ⟨op c, congr_arg op hc⟩⟩


@[to_additive]
lemma isSquare_unop_iff {a : αᵐᵒᵖ} : IsSquare (unop a) ↔ IsSquare a := isSquare_op_iff.symm


@[to_additive]
instance [DecidablePred (IsSquare : α → Prop)] : DecidablePred (IsSquare : αᵐᵒᵖ → Prop) :=
  fun _ ↦ decidable_of_iff _ isSquare_unop_iff


@[simp]
lemma even_ofMul_iff {a : α} : Even (Additive.ofMul a) ↔ IsSquare a := Iff.rfl


@[simp]
lemma isSquare_toMul_iff {a : Additive α} : IsSquare (a.toMul) ↔ Even a := Iff.rfl


instance Additive.instDecidablePredEven [DecidablePred (IsSquare : α → Prop)] :
    DecidablePred (Even : Additive α → Prop) :=
  fun _ ↦ decidable_of_iff _ isSquare_toMul_iff


@[simp] lemma isSquare_ofAdd_iff {a : α} : IsSquare (Multiplicative.ofAdd a) ↔ Even a := Iff.rfl


@[simp]
lemma even_toAdd_iff {a : Multiplicative α} : Even a.toAdd ↔ IsSquare a := Iff.rfl


instance Multiplicative.instDecidablePredIsSquare [DecidablePred (Even : α → Prop)] :
    DecidablePred (IsSquare : Multiplicative α → Prop) :=
  fun _ ↦ decidable_of_iff _ even_toAdd_iff


@[to_additive (attr := simp)]
lemma IsSquare.one [MulOneClass α] : IsSquare (1 : α) := ⟨1, (mul_one _).symm⟩


@[to_additive, deprecated (since := "2024-12-27")] alias isSquare_one := IsSquare.one


@[to_additive]
lemma IsSquare.map [MulOneClass α] [MulOneClass β] [FunLike F α β] [MonoidHomClass F α β]
    {m : α} (f : F) :
    IsSquare m → IsSquare (f m) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : MulOneClass α
    inst✝² : MulOneClass β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : α
    f : F
    ⊢ IsSquare m → IsSquare (f m)
  -/
  rintro ⟨m, rfl⟩
  /-
    case intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : MulOneClass α
    inst✝² : MulOneClass β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    f : F
    m : α
    ⊢ IsSquare (f (HMul.hMul m m))
  -/
  exact ⟨f m, by simp⟩
  /-
    🎉 no goals
  -/


@[to_additive even_iff_exists_two_nsmul]
                                                                         /-
                                                                           α : Type u_2
                                                                           inst✝ : Monoid α
                                                                           m : α
                                                                           ⊢ Iff (IsSquare m) (Exists fun c => Eq m (HPow.hPow c 2))
                                                                         -/
lemma isSquare_iff_exists_sq (m : α) : IsSquare m ↔ ∃ c, m = c ^ 2 := by simp [IsSquare, pow_two]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


alias ⟨IsSquare.exists_sq, isSquare_of_exists_sq⟩ := isSquare_iff_exists_sq


attribute [to_additive Even.exists_two_nsmul
  "Alias of the forwards direction of `even_iff_exists_two_nsmul`."] IsSquare.exists_sq


@[to_additive] lemma IsSquare.pow (n : ℕ) : IsSquare a → IsSquare (a ^ n) := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    a : α
    n : Nat
    ⊢ IsSquare a → IsSquare (HPow.hPow a n)
  -/
  rintro ⟨a, rfl⟩; exact ⟨a ^ n, (Commute.refl _).mul_pow _⟩
                   /-
                     🎉 no goals
                   -/


@[to_additive Even.nsmul'] lemma Even.isSquare_pow : Even n → ∀ a : α, IsSquare (a ^ n) := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    n : Nat
    ⊢ Even n → ∀ (a : α), IsSquare (HPow.hPow a n)
  -/
  rintro ⟨n, rfl⟩ a; exact ⟨a ^ n, pow_add _ _ _⟩
                     /-
                       🎉 no goals
                     -/


@[to_additive Even.two_nsmul] lemma IsSquare.sq (a : α) : IsSquare (a ^ 2) := ⟨a, pow_two _⟩


@[deprecated (since := "2024-12-27")] alias IsSquare_sq := IsSquare.sq

@[deprecated (since := "2024-12-27")] alias even_two_nsmul := Even.two_nsmul


@[to_additive]
lemma IsSquare.mul [CommSemigroup α] {a b : α} : IsSquare a → IsSquare b → IsSquare (a * b) := by
  /-
    α : Type u_2
    inst✝ : CommSemigroup α
    a b : α
    ⊢ IsSquare a → IsSquare b → IsSquare (HMul.hMul a b)
  -/
  rintro ⟨a, rfl⟩ ⟨b, rfl⟩; exact ⟨a * b, mul_mul_mul_comm _ _ _ _⟩
                            /-
                              🎉 no goals
                            -/


@[to_additive (attr := simp)] lemma isSquare_inv : IsSquare a⁻¹ ↔ IsSquare a := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    a : α
    ⊢ Iff (IsSquare (Inv.inv a)) (IsSquare a)
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      a : α
      h : IsSquare (Inv.inv a)
      ⊢ IsSquare a
    -/
  · rw [← isSquare_op_iff, ← inv_inv a]
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      a : α
      h : IsSquare (Inv.inv a)
      ⊢ IsSquare (MulOpposite.op (Inv.inv (Inv.inv a)))
    -/
    exact h.map (MulEquiv.inv' α)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : DivisionMonoid α
      a : α
      h : IsSquare a
      ⊢ IsSquare (Inv.inv a)
    -/
  · exact (isSquare_op_iff.mpr h).map (MulEquiv.inv' α).symm
    /-
      🎉 no goals
    -/


alias ⟨_, IsSquare.inv⟩ := isSquare_inv


attribute [to_additive] IsSquare.inv


@[to_additive] lemma IsSquare.zpow (n : ℤ) : IsSquare a → IsSquare (a ^ n) := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    a : α
    n : Int
    ⊢ IsSquare a → IsSquare (HPow.hPow a n)
  -/
  rintro ⟨a, rfl⟩; exact ⟨a ^ n, (Commute.refl _).mul_zpow _⟩
                   /-
                     🎉 no goals
                   -/


@[to_additive]
lemma IsSquare.div [DivisionCommMonoid α] {a b : α} (ha : IsSquare a) (hb : IsSquare b) :
                           /-
                             α : Type u_2
                             inst✝ : DivisionCommMonoid α
                             a b : α
                             ha : IsSquare a
                             hb : IsSquare b
                             ⊢ IsSquare (HDiv.hDiv a b)
                           -/
    IsSquare (a / b) := by rw [div_eq_mul_inv]; exact ha.mul hb.inv
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp) Even.zsmul']
lemma Even.isSquare_zpow [Group α] {n : ℤ} : Even n → ∀ a : α, IsSquare (a ^ n) := by
  /-
    α : Type u_2
    inst✝ : Group α
    n : Int
    ⊢ Even n → ∀ (a : α), IsSquare (HPow.hPow a n)
  -/
  rintro ⟨n, rfl⟩ a; exact ⟨a ^ n, zpow_add _ _ _⟩
                     /-
                       🎉 no goals
                     -/

