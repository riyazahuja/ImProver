@[to_additive (attr := simp) dite_smul]
lemma pow_dite (p : Prop) [Decidable p] (a : α) (b : p → β) (c : ¬ p → β) :
                                                                                /-
                                                                                  α : Type u_1
                                                                                  β : Type u_2
                                                                                  inst✝¹ : Pow α β
                                                                                  p : Prop
                                                                                  inst✝ : Decidable p
                                                                                  a : α
                                                                                  b : p → β
                                                                                  c : Not p → β
                                                                                  ⊢ Eq (HPow.hPow a (dite p (fun h => b h) fun h => c h)) (dite p (fun h => HPow …
                                                                                -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
    a ^ (if h : p then b h else c h) = if h : p then a ^ b h else a ^ c h := by split_ifs <;> rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[to_additive (attr := simp) smul_dite]
lemma dite_pow (p : Prop) [Decidable p] (a : p → α) (b : ¬ p → α) (c : β) :
                                                                                /-
                                                                                  α : Type u_1
                                                                                  β : Type u_2
                                                                                  inst✝¹ : Pow α β
                                                                                  p : Prop
                                                                                  inst✝ : Decidable p
                                                                                  a : p → α
                                                                                  b : Not p → α
                                                                                  c : β
                                                                                  ⊢ Eq (HPow.hPow (dite p (fun h => a h) fun h => b h) c) (dite p (fun h => HPow …
                                                                                -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
    (if h : p then a h else b h) ^ c = if h : p then a h ^ c else b h ^ c := by split_ifs <;> rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[to_additive (attr := simp) ite_smul]
lemma pow_ite (p : Prop) [Decidable p] (a : α) (b c : β) :
    a ^ (if p then b else c) = if p then a ^ b else a ^ c := pow_dite _ _ _ _


@[to_additive (attr := simp) smul_ite]
lemma ite_pow (p : Prop) [Decidable p] (a b : α) (c : β) :
    (if p then a else b) ^ c = if p then a ^ c else b ^ c := dite_pow _ _ _ _


set_option linter.existingAttributeWarning false in
attribute [to_additive (attr := simp)] dite_smul smul_dite ite_smul smul_ite


@[to_additive]
theorem mul_right_injective (a : G) : Injective (a * ·) := fun _ _ ↦ mul_left_cancel


@[to_additive (attr := simp)]
theorem mul_right_inj (a : G) {b c : G} : a * b = a * c ↔ b = c :=
  (mul_right_injective a).eq_iff


@[to_additive]
theorem mul_ne_mul_right (a : G) {b c : G} : a * b ≠ a * c ↔ b ≠ c :=
  (mul_right_injective a).ne_iff


@[to_additive]
theorem mul_left_injective (a : G) : Function.Injective (· * a) := fun _ _ ↦ mul_right_cancel


@[to_additive (attr := simp)]
theorem mul_left_inj (a : G) {b c : G} : b * a = c * a ↔ b = c :=
  (mul_left_injective a).eq_iff


@[to_additive]
theorem mul_ne_mul_left (a : G) {b c : G} : b * a ≠ c * a ↔ b ≠ c :=
  (mul_left_injective a).ne_iff


@[to_additive]
instance Semigroup.to_isAssociative : Std.Associative (α := α) (· * ·) := ⟨mul_assoc⟩


/-- Composing two multiplications on the left by `y` then `x`
is equal to a multiplication on the left by `x * y`.
-/
@[to_additive (attr := simp) "Composing two additions on the left by `y` then `x`
is equal to an addition on the left by `x + y`."]
theorem comp_mul_left (x y : α) : (x * ·) ∘ (y * ·) = (x * y * ·) := by
  /-
    α : Type u_1
    inst✝ : Semigroup α
    x y : α
    ⊢ Eq (Function.comp (fun x_1 => HMul.hMul x x_1) fun x => HMul.hMul y x) fun x …
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝ : Semigroup α
    x y z : α
    ⊢ Eq (Function.comp (fun x_1 => HMul.hMul x x_1) (fun x => HMul.hMul y x) z) ( …
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/


/-- Composing two multiplications on the right by `y` and `x`
is equal to a multiplication on the right by `y * x`.
-/
@[to_additive (attr := simp) "Composing two additions on the right by `y` and `x`
is equal to an addition on the right by `y + x`."]
theorem comp_mul_right (x y : α) : (· * x) ∘ (· * y) = (· * (y * x)) := by
  /-
    α : Type u_1
    inst✝ : Semigroup α
    x y : α
    ⊢ Eq (Function.comp (fun x_1 => HMul.hMul x_1 x) fun x => HMul.hMul x y) fun x …
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝ : Semigroup α
    x y z : α
    ⊢ Eq (Function.comp (fun x_1 => HMul.hMul x_1 x) (fun x => HMul.hMul x y) z) ( …
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive]
instance CommMagma.to_isCommutative [CommMagma G] : Std.Commutative (α := G) (· * ·) := ⟨mul_comm⟩


@[to_additive]
theorem ite_mul_one {P : Prop} [Decidable P] {a b : M} :
    ite P (a * b) 1 = ite P a 1 * ite P b 1 := by
  /-
    M : Type u_4
    inst✝¹ : MulOneClass M
    P : Prop
    inst✝ : Decidable P
    a b : M
    ⊢ Eq (ite P (HMul.hMul a b) 1) (HMul.hMul (ite P a 1) (ite P b 1))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : P <;> simp [h]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem ite_one_mul {P : Prop} [Decidable P] {a b : M} :
    ite P 1 (a * b) = ite P 1 a * ite P 1 b := by
  /-
    M : Type u_4
    inst✝¹ : MulOneClass M
    P : Prop
    inst✝ : Decidable P
    a b : M
    ⊢ Eq (ite P 1 (HMul.hMul a b)) (HMul.hMul (ite P 1 a) (ite P 1 b))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : P <;> simp [h]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem eq_one_iff_eq_one_of_mul_eq_one {a b : M} (h : a * b = 1) : a = 1 ↔ b = 1 := by
  /-
    M : Type u_4
    inst✝ : MulOneClass M
    a b : M
    h : Eq (HMul.hMul a b) 1
    ⊢ Iff (Eq a 1) (Eq b 1)
  -/
                               /-
                                 🎉 no goals
                               -/
  constructor <;> (rintro rfl; simpa using h)
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
theorem one_mul_eq_id : ((1 : M) * ·) = id :=
  funext one_mul


@[to_additive]
theorem mul_one_eq_id : (· * (1 : M)) = id :=
  funext mul_one


@[to_additive]
theorem mul_left_comm (a b c : G) : a * (b * c) = b * (a * c) := by
  /-
    G : Type u_3
    inst✝ : CommSemigroup G
    a b c : G
    ⊢ Eq (HMul.hMul a (HMul.hMul b c)) (HMul.hMul b (HMul.hMul a c))
  -/
  rw [← mul_assoc, mul_comm a, mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_right_comm (a b c : G) : a * b * c = a * c * b := by
  /-
    G : Type u_3
    inst✝ : CommSemigroup G
    a b c : G
    ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul (HMul.hMul a c) b)
  -/
  rw [mul_assoc, mul_comm b, mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mul_mul_comm (a b c d : G) : a * b * (c * d) = a * c * (b * d) := by
  /-
    G : Type u_3
    inst✝ : CommSemigroup G
    a b c d : G
    ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HMul.hMul a c) (H …
  -/
  simp only [mul_left_comm, mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_rotate (a b c : G) : a * b * c = b * c * a := by
  /-
    G : Type u_3
    inst✝ : CommSemigroup G
    a b c : G
    ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul (HMul.hMul b c) a)
  -/
  simp only [mul_left_comm, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_rotate' (a b c : G) : a * (b * c) = b * (c * a) := by
  /-
    G : Type u_3
    inst✝ : CommSemigroup G
    a b c : G
    ⊢ Eq (HMul.hMul a (HMul.hMul b c)) (HMul.hMul b (HMul.hMul c a))
  -/
  simp only [mul_left_comm, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive boole_nsmul]
lemma pow_boole (P : Prop) [Decidable P] (a : M) :
                                                        /-
                                                          M : Type u_4
                                                          inst✝¹ : Monoid M
                                                          P : Prop
                                                          inst✝ : Decidable P
                                                          a : M
                                                          ⊢ Eq (HPow.hPow a (ite P 1 0)) (ite P a 1)
                                                        -/
    (a ^ if P then 1 else 0) = if P then a else 1 := by simp only [pow_ite, pow_one, pow_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive nsmul_add_sub_nsmul]
lemma pow_mul_pow_sub (a : M) (h : m ≤ n) : a ^ m * a ^ (n - m) = a ^ n := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    m n : Nat
    a : M
    h : LE.le m n
    ⊢ Eq (HMul.hMul (HPow.hPow a m) (HPow.hPow a (HSub.hSub n m))) (HPow.hPow a n)
  -/
  rw [← pow_add, Nat.add_comm, Nat.sub_add_cancel h]
  /-
    🎉 no goals
  -/


@[to_additive sub_nsmul_nsmul_add]
lemma pow_sub_mul_pow (a : M) (h : m ≤ n) : a ^ (n - m) * a ^ m = a ^ n := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    m n : Nat
    a : M
    h : LE.le m n
    ⊢ Eq (HMul.hMul (HPow.hPow a (HSub.hSub n m)) (HPow.hPow a m)) (HPow.hPow a n)
  -/
  rw [← pow_add, Nat.sub_add_cancel h]
  /-
    🎉 no goals
  -/


@[to_additive sub_one_nsmul_add]
lemma mul_pow_sub_one (hn : n ≠ 0) (a : M) : a * a ^ (n - 1) = a ^ n := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    n : Nat
    hn : Ne n 0
    a : M
    ⊢ Eq (HMul.hMul a (HPow.hPow a (HSub.hSub n 1))) (HPow.hPow a n)
  -/
  rw [← pow_succ', Nat.sub_add_cancel <| Nat.one_le_iff_ne_zero.2 hn]
  /-
    🎉 no goals
  -/


@[to_additive add_sub_one_nsmul]
lemma pow_sub_one_mul (hn : n ≠ 0) (a : M) : a ^ (n - 1) * a = a ^ n := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    n : Nat
    hn : Ne n 0
    a : M
    ⊢ Eq (HMul.hMul (HPow.hPow a (HSub.hSub n 1)) a) (HPow.hPow a n)
  -/
  rw [← pow_succ, Nat.sub_add_cancel <| Nat.one_le_iff_ne_zero.2 hn]
  /-
    🎉 no goals
  -/


/-- If `x ^ n = 1`, then `x ^ m` is the same as `x ^ (m % n)` -/
@[to_additive nsmul_eq_mod_nsmul "If `n • x = 0`, then `m • x` is the same as `(m % n) • x`"]
lemma pow_eq_pow_mod (m : ℕ) (ha : a ^ n = 1) : a ^ m = a ^ (m % n) := by
  calc
    a ^ m = a ^ (m % n + n * (m / n)) := by rw [Nat.mod_add_div]
    _ = a ^ (m % n) := by simp [pow_add, pow_mul, ha]


@[to_additive] lemma pow_mul_pow_eq_one : ∀ n, a * b = 1 → a ^ n * b ^ n = 1
               /-
                 M : Type u_4
                 inst✝ : Monoid M
                 a b : M
                 x✝ : Eq (HMul.hMul a b) 1
                 ⊢ Eq (HMul.hMul (HPow.hPow a 0) (HPow.hPow b 0)) 1
               -/
  | 0, _ => by simp
               /-
                 🎉 no goals
               -/
  | n + 1, h =>
    calc
                                                              /-
                                                                M : Type u_4
                                                                inst✝ : Monoid M
                                                                a b : M
                                                                n : Nat
                                                                h : Eq (HMul.hMul a b) 1
                                                                ⊢ Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow b n.succ)) (HMul.hMul (HMul.hM …
                                                              -/
      a ^ n.succ * b ^ n.succ = a ^ n * a * (b * b ^ n) := by rw [pow_succ, pow_succ']
                                                              /-
                                                                🎉 no goals
                                                              -/
                                        /-
                                          M : Type u_4
                                          inst✝ : Monoid M
                                          a b : M
                                          n : Nat
                                          h : Eq (HMul.hMul a b) 1
                                          ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a n) a) (HMul.hMul b (HPow.hPow b n))) ( …
                                        -/
      _ = a ^ n * (a * b) * b ^ n := by simp only [mul_assoc]
                                        /-
                                          🎉 no goals
                                        -/
                  /-
                    M : Type u_4
                    inst✝ : Monoid M
                    a b : M
                    n : Nat
                    h : Eq (HMul.hMul a b) 1
                    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a n) (HMul.hMul a b)) (HPow.hPow b n)) 1
                  -/
      _ = 1 := by simp [h, pow_mul_pow_eq_one]
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem inv_unique (hy : x * y = 1) (hz : x * z = 1) : y = z :=
  left_inv_eq_right_inv (Trans.trans (mul_comm _ _) hy) hz


@[to_additive nsmul_add] lemma mul_pow (a b : M) : ∀ n, (a * b) ^ n = a ^ n * b ^ n
            /-
              M : Type u_4
              inst✝ : CommMonoid M
              a b : M
              ⊢ Eq (HPow.hPow (HMul.hMul a b) 0) (HMul.hMul (HPow.hPow a 0) (HPow.hPow b 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, pow_zero, one_mul]
            /-
              🎉 no goals
            -/
                /-
                  M : Type u_4
                  inst✝ : CommMonoid M
                  a b : M
                  n : Nat
                  ⊢ Eq (HPow.hPow (HMul.hMul a b) (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow a (HAdd …
                -/
  | n + 1 => by rw [pow_succ', pow_succ', pow_succ', mul_pow, mul_mul_mul_comm]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
theorem mul_right_eq_self : a * b = a ↔ b = 1 := calc
                                  /-
                                    M : Type u_4
                                    inst✝ : LeftCancelMonoid M
                                    a b : M
                                    ⊢ Iff (Eq (HMul.hMul a b) a) (Eq (HMul.hMul a b) (HMul.hMul a 1))
                                  -/
  a * b = a ↔ a * b = a * 1 := by rw [mul_one]
                                  /-
                                    🎉 no goals
                                  -/
  _ ↔ b = 1 := mul_left_cancel_iff


@[to_additive (attr := simp)]
theorem self_eq_mul_right : a = a * b ↔ b = 1 :=
  eq_comm.trans mul_right_eq_self


@[to_additive]
theorem mul_right_ne_self : a * b ≠ a ↔ b ≠ 1 := mul_right_eq_self.not


@[to_additive]
theorem self_ne_mul_right : a ≠ a * b ↔ b ≠ 1 := self_eq_mul_right.not


@[to_additive (attr := simp)]
theorem mul_left_eq_self : a * b = b ↔ a = 1 := calc
                                  /-
                                    M : Type u_4
                                    inst✝ : RightCancelMonoid M
                                    a b : M
                                    ⊢ Iff (Eq (HMul.hMul a b) b) (Eq (HMul.hMul a b) (HMul.hMul 1 b))
                                  -/
  a * b = b ↔ a * b = 1 * b := by rw [one_mul]
                                  /-
                                    🎉 no goals
                                  -/
  _ ↔ a = 1 := mul_right_cancel_iff


@[to_additive (attr := simp)]
theorem self_eq_mul_left : b = a * b ↔ a = 1 :=
  eq_comm.trans mul_left_eq_self


@[to_additive]
theorem mul_left_ne_self : a * b ≠ b ↔ a ≠ 1 := mul_left_eq_self.not


@[to_additive]
theorem self_ne_mul_left : b ≠ a * b ↔ a ≠ 1 := self_eq_mul_left.not


                                                                                       /-
                                                                                         α : Type u_1
                                                                                         inst✝ : CancelCommMonoid α
                                                                                         a b c d : α
                                                                                         h : Eq (HMul.hMul a b) (HMul.hMul c d)
                                                                                         ⊢ Iff (Eq a c) (Eq b d)
                                                                                       -/
@[to_additive] lemma eq_iff_eq_of_mul_eq_mul (h : a * b = c * d) : a = c ↔ b = d := by aesop
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/

                                                                                       /-
                                                                                         α : Type u_1
                                                                                         inst✝ : CancelCommMonoid α
                                                                                         a b c d : α
                                                                                         h : Eq (HMul.hMul a b) (HMul.hMul c d)
                                                                                         ⊢ Iff (Ne a c) (Ne b d)
                                                                                       -/
@[to_additive] lemma ne_iff_ne_of_mul_eq_mul (h : a * b = c * d) : a ≠ c ↔ b ≠ d := by aesop
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[to_additive (attr := simp)]
theorem inv_involutive : Function.Involutive (Inv.inv : G → G) :=
  inv_inv


@[to_additive (attr := simp)]
theorem inv_surjective : Function.Surjective (Inv.inv : G → G) :=
  inv_involutive.surjective


@[to_additive]
theorem inv_injective : Function.Injective (Inv.inv : G → G) :=
  inv_involutive.injective


@[to_additive (attr := simp)]
theorem inv_inj : a⁻¹ = b⁻¹ ↔ a = b :=
  inv_injective.eq_iff


@[to_additive]
theorem inv_eq_iff_eq_inv : a⁻¹ = b ↔ a = b⁻¹ :=
  ⟨fun h => h ▸ (inv_inv a).symm, fun h => h.symm ▸ inv_inv b⟩


@[to_additive]
theorem inv_comp_inv : Inv.inv ∘ Inv.inv = @id G :=
  inv_involutive.comp_self


@[to_additive]
theorem leftInverse_inv : LeftInverse (fun a : G ↦ a⁻¹) fun a ↦ a⁻¹ :=
  inv_inv


@[to_additive]
theorem rightInverse_inv : RightInverse (fun a : G ↦ a⁻¹) fun a ↦ a⁻¹ :=
  inv_inv


@[to_additive, field_simps] -- The attributes are out of order on purpose
                                                   /-
                                                     G : Type u_3
                                                     inst✝ : DivInvMonoid G
                                                     x : G
                                                     ⊢ Eq (Inv.inv x) (HDiv.hDiv 1 x)
                                                   -/
theorem inv_eq_one_div (x : G) : x⁻¹ = 1 / x := by rw [div_eq_mul_inv, one_mul]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
theorem mul_one_div (x y : G) : x * (1 / y) = x / y := by
  /-
    G : Type u_3
    inst✝ : DivInvMonoid G
    x y : G
    ⊢ Eq (HMul.hMul x (HDiv.hDiv 1 y)) (HDiv.hDiv x y)
  -/
  rw [div_eq_mul_inv, one_mul, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_div_assoc (a b c : G) : a * b / c = a * (b / c) := by
  /-
    G : Type u_3
    inst✝ : DivInvMonoid G
    a b c : G
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) c) (HMul.hMul a (HDiv.hDiv b c))
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, mul_assoc _ _ _]
  /-
    🎉 no goals
  -/


@[to_additive, field_simps] -- The attributes are out of order on purpose
theorem mul_div_assoc' (a b c : G) : a * (b / c) = a * b / c :=
  (mul_div_assoc _ _ _).symm


@[to_additive (attr := simp)]
theorem one_div (a : G) : 1 / a = a⁻¹ :=
  (inv_eq_one_div a).symm


@[to_additive]
                                                            /-
                                                              G : Type u_3
                                                              inst✝ : DivInvMonoid G
                                                              a b c : G
                                                              ⊢ Eq (HMul.hMul a (HDiv.hDiv b c)) (HDiv.hDiv (HMul.hMul a b) c)
                                                            -/
theorem mul_div (a b c : G) : a * (b / c) = a * b / c := by simp only [mul_assoc, div_eq_mul_inv]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : DivInvMonoid G
                                                                   a b : G
                                                                   ⊢ Eq (HDiv.hDiv a b) (HMul.hMul a (HDiv.hDiv 1 b))
                                                                 -/
theorem div_eq_mul_one_div (a b : G) : a / b = a * (1 / b) := by rw [div_eq_mul_inv, one_div]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive (attr := simp)]
                                          /-
                                            G : Type u_3
                                            inst✝ : DivInvOneMonoid G
                                            a : G
                                            ⊢ Eq (HDiv.hDiv a 1) a
                                          -/
theorem div_one (a : G) : a / 1 = a := by simp [div_eq_mul_inv]
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem one_div_one : (1 : G) / 1 = 1 :=
  div_one _


@[to_additive]
theorem eq_inv_of_mul_eq_one_right (h : a * b = 1) : b = a⁻¹ :=
  (inv_eq_of_mul_eq_one_right h).symm


@[to_additive]
theorem eq_one_div_of_mul_eq_one_left (h : b * a = 1) : b = 1 / a := by
  /-
    α : Type u_1
    inst✝ : DivisionMonoid α
    a b : α
    h : Eq (HMul.hMul b a) 1
    ⊢ Eq b (HDiv.hDiv 1 a)
  -/
  rw [eq_inv_of_mul_eq_one_left h, one_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_one_div_of_mul_eq_one_right (h : a * b = 1) : b = 1 / a := by
  /-
    α : Type u_1
    inst✝ : DivisionMonoid α
    a b : α
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq b (HDiv.hDiv 1 a)
  -/
  rw [eq_inv_of_mul_eq_one_right h, one_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_of_div_eq_one (h : a / b = 1) : a = b :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : DivisionMonoid α
                                                      a b : α
                                                      h : Eq (HDiv.hDiv a b) 1
                                                      ⊢ Eq (HMul.hMul a (Inv.inv b)) 1
                                                    -/
  inv_injective <| inv_eq_of_mul_eq_one_right <| by rwa [← div_eq_mul_inv]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive]
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DivisionMonoid α
                                                             a b : α
                                                             h : Eq (HMul.hMul (Inv.inv a) b) 1
                                                             ⊢ Eq a b
                                                           -/
lemma eq_of_inv_mul_eq_one (h : a⁻¹ * b = 1) : a = b := by simpa using eq_inv_of_mul_eq_one_left h
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DivisionMonoid α
                                                             a b : α
                                                             h : Eq (HMul.hMul a (Inv.inv b)) 1
                                                             ⊢ Eq a b
                                                           -/
lemma eq_of_mul_inv_eq_one (h : a * b⁻¹ = 1) : a = b := by simpa using eq_inv_of_mul_eq_one_left h
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
theorem div_ne_one_of_ne : a ≠ b → a / b ≠ 1 :=
  mt eq_of_div_eq_one


@[to_additive]
                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : DivisionMonoid α
                                                                        a b : α
                                                                        ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) (HDiv.hDiv 1 b)) (HDiv.hDiv 1 (HMul.hMul b a))
                                                                      -/
theorem one_div_mul_one_div_rev : 1 / a * (1 / b) = 1 / (b * a) := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive]
                                                 /-
                                                   α : Type u_1
                                                   inst✝ : DivisionMonoid α
                                                   a b : α
                                                   ⊢ Eq (HDiv.hDiv (Inv.inv a) b) (Inv.inv (HMul.hMul b a))
                                                 -/
theorem inv_div_left : a⁻¹ / b = (b * a)⁻¹ := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive (attr := simp)]
                                          /-
                                            α : Type u_1
                                            inst✝ : DivisionMonoid α
                                            a b : α
                                            ⊢ Eq (Inv.inv (HDiv.hDiv a b)) (HDiv.hDiv b a)
                                          -/
theorem inv_div : (a / b)⁻¹ = b / a := by simp
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
                                                /-
                                                  α : Type u_1
                                                  inst✝ : DivisionMonoid α
                                                  a b : α
                                                  ⊢ Eq (HDiv.hDiv 1 (HDiv.hDiv a b)) (HDiv.hDiv b a)
                                                -/
theorem one_div_div : 1 / (a / b) = b / a := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
                                                /-
                                                  α : Type u_1
                                                  inst✝ : DivisionMonoid α
                                                  a : α
                                                  ⊢ Eq (HDiv.hDiv 1 (HDiv.hDiv 1 a)) a
                                                -/
theorem one_div_one_div : 1 / (1 / a) = a := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem div_eq_div_iff_comm : a / b = c / d ↔ b / a = d / c :=
                           /-
                             α : Type u_1
                             inst✝ : DivisionMonoid α
                             a b c d : α
                             ⊢ Iff (Eq (Inv.inv (HDiv.hDiv a b)) (Inv.inv (HDiv.hDiv c d))) (Eq (HDiv.hDiv  …
                           -/
  inv_inj.symm.trans <| by simp only [inv_div]
                           /-
                             🎉 no goals
                           -/


@[to_additive]
instance (priority := 100) DivisionMonoid.toDivInvOneMonoid : DivInvOneMonoid α :=
  { DivisionMonoid.toDivInvMonoid with
                  /-
                    α : Type u_1
                    β : Type u_2
                    G : Type u_3
                    M : Type u_4
                    inst✝ : DivisionMonoid α
                    a b c d : α
                    ⊢ Eq (Inv.inv 1) 1
                  -/
    inv_one := by simpa only [one_div, inv_inv] using (inv_div (1 : α) 1).symm }
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
lemma inv_pow (a : α) : ∀ n : ℕ, a⁻¹ ^ n = (a ^ n)⁻¹
            /-
              α : Type u_1
              inst✝ : DivisionMonoid α
              a : α
              ⊢ Eq (HPow.hPow (Inv.inv a) 0) (Inv.inv (HPow.hPow a 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, inv_one]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  inst✝ : DivisionMonoid α
                  a : α
                  n : Nat
                  ⊢ Eq (HPow.hPow (Inv.inv a) (HAdd.hAdd n 1)) (Inv.inv (HPow.hPow a (HAdd.hAdd  …
                -/
  | n + 1 => by rw [pow_succ', pow_succ, inv_pow _ n, mul_inv_rev]
                /-
                  🎉 no goals
                -/

-- the attributes are intentionally out of order. `smul_zero` proves `zsmul_zero`.

@[to_additive zsmul_zero, simp]
lemma one_zpow : ∀ n : ℤ, (1 : α) ^ n = 1
                     /-
                       α : Type u_1
                       inst✝ : DivisionMonoid α
                       n : Nat
                       ⊢ Eq (HPow.hPow 1 ↑n) 1
                     -/
  | (n : ℕ)    => by rw [zpow_natCast, one_pow]
                     /-
                       🎉 no goals
                     -/
                     /-
                       α : Type u_1
                       inst✝ : DivisionMonoid α
                       n : Nat
                       ⊢ Eq (HPow.hPow 1 (Int.negSucc n)) 1
                     -/
  | .negSucc n => by rw [zpow_negSucc, one_pow, inv_one]
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp) neg_zsmul]
lemma zpow_neg (a : α) : ∀ n : ℤ, a ^ (-n) = (a ^ n)⁻¹
  | (_ + 1 : ℕ) => DivInvMonoid.zpow_neg' _ _
  | 0 => by
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      ⊢ Eq (HPow.hPow a (-0)) (Inv.inv (HPow.hPow a 0))
    -/
    change a ^ (0 : ℤ) = (a ^ (0 : ℤ))⁻¹
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      ⊢ Eq (HPow.hPow a 0) (Inv.inv (HPow.hPow a 0))
    -/
    simp
    /-
      🎉 no goals
    -/
  | Int.negSucc n => by
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      n : Nat
      ⊢ Eq (HPow.hPow a (Neg.neg (Int.negSucc n))) (Inv.inv (HPow.hPow a (Int.negSuc …
    -/
    rw [zpow_negSucc, inv_inv, ← zpow_natCast]
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      n : Nat
      ⊢ Eq (HPow.hPow a (Neg.neg (Int.negSucc n))) (HPow.hPow a ↑(HAdd.hAdd n 1))
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive neg_one_zsmul_add]
lemma mul_zpow_neg_one (a b : α) : (a * b) ^ (-1 : ℤ) = b ^ (-1 : ℤ) * a ^ (-1 : ℤ) := by
  /-
    α : Type u_1
    inst✝ : DivisionMonoid α
    a b : α
    ⊢ Eq (HPow.hPow (HMul.hMul a b) (-1)) (HMul.hMul (HPow.hPow b (-1)) (HPow.hPow …
  -/
  simp only [zpow_neg, zpow_one, mul_inv_rev]
  /-
    🎉 no goals
  -/


@[to_additive zsmul_neg]
lemma inv_zpow (a : α) : ∀ n : ℤ, a⁻¹ ^ n = (a ^ n)⁻¹
                     /-
                       α : Type u_1
                       inst✝ : DivisionMonoid α
                       a : α
                       n : Nat
                       ⊢ Eq (HPow.hPow (Inv.inv a) ↑n) (Inv.inv (HPow.hPow a ↑n))
                     -/
  | (n : ℕ)    => by rw [zpow_natCast, zpow_natCast, inv_pow]
                     /-
                       🎉 no goals
                     -/
                     /-
                       α : Type u_1
                       inst✝ : DivisionMonoid α
                       a : α
                       n : Nat
                       ⊢ Eq (HPow.hPow (Inv.inv a) (Int.negSucc n)) (Inv.inv (HPow.hPow a (Int.negSuc …
                     -/
  | .negSucc n => by rw [zpow_negSucc, zpow_negSucc, inv_pow]
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp) zsmul_neg']
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DivisionMonoid α
                                                             a : α
                                                             n : Int
                                                             ⊢ Eq (HPow.hPow (Inv.inv a) n) (HPow.hPow a (Neg.neg n))
                                                           -/
lemma inv_zpow' (a : α) (n : ℤ) : a⁻¹ ^ n = a ^ (-n) := by rw [inv_zpow, zpow_neg]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive nsmul_zero_sub]
                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : DivisionMonoid α
                                                                    a : α
                                                                    n : Nat
                                                                    ⊢ Eq (HPow.hPow (HDiv.hDiv 1 a) n) (HDiv.hDiv 1 (HPow.hPow a n))
                                                                  -/
lemma one_div_pow (a : α) (n : ℕ) : (1 / a) ^ n = 1 / a ^ n := by simp only [one_div, inv_pow]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive zsmul_zero_sub]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DivisionMonoid α
                                                                     a : α
                                                                     n : Int
                                                                     ⊢ Eq (HPow.hPow (HDiv.hDiv 1 a) n) (HDiv.hDiv 1 (HPow.hPow a n))
                                                                   -/
lemma one_div_zpow (a : α) (n : ℤ) : (1 / a) ^ n = 1 / a ^ n := by simp only [one_div, inv_zpow]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive (attr := simp)]
theorem inv_eq_one : a⁻¹ = 1 ↔ a = 1 :=
  inv_injective.eq_iff' inv_one


@[to_additive (attr := simp)]
theorem one_eq_inv : 1 = a⁻¹ ↔ a = 1 :=
  eq_comm.trans inv_eq_one


@[to_additive]
theorem inv_ne_one : a⁻¹ ≠ 1 ↔ a ≠ 1 :=
  inv_eq_one.not


@[to_additive]
theorem eq_of_one_div_eq_one_div (h : 1 / a = 1 / b) : a = b := by
  /-
    α : Type u_1
    inst✝ : DivisionMonoid α
    a b : α
    h : Eq (HDiv.hDiv 1 a) (HDiv.hDiv 1 b)
    ⊢ Eq a b
  -/
  rw [← one_div_one_div a, h, one_div_one_div]
  /-
    🎉 no goals
  -/

-- Note that `mul_zsmul` and `zpow_mul` have the primes swapped
-- when additivised since their argument order,
-- and therefore the more "natural" choice of lemma, is reversed.

@[to_additive mul_zsmul'] lemma zpow_mul (a : α) : ∀ m n : ℤ, a ^ (m * n) = (a ^ m) ^ n
  | (m : ℕ), (n : ℕ) => by
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      m n : Nat
      ⊢ Eq (HPow.hPow a (HMul.hMul ↑m ↑n)) (HPow.hPow (HPow.hPow a ↑m) ↑n)
    -/
    rw [zpow_natCast, zpow_natCast, ← pow_mul, ← zpow_natCast]
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      m n : Nat
      ⊢ Eq (HPow.hPow a (HMul.hMul ↑m ↑n)) (HPow.hPow a ↑(HMul.hMul m n))
    -/
    rfl
    /-
      🎉 no goals
    -/
  | (m : ℕ), .negSucc n => by
    rw [zpow_natCast, zpow_negSucc, ← pow_mul, Int.ofNat_mul_negSucc, zpow_neg, inv_inj,
      ← zpow_natCast]
  | .negSucc m, (n : ℕ) => by
    rw [zpow_natCast, zpow_negSucc, ← inv_pow, ← pow_mul, Int.negSucc_mul_ofNat, zpow_neg, inv_pow,
      inv_inj, ← zpow_natCast]
  | .negSucc m, .negSucc n => by
    rw [zpow_negSucc, zpow_negSucc, Int.negSucc_mul_negSucc, inv_pow, inv_inv, ← pow_mul, ←
      zpow_natCast]
    /-
      α : Type u_1
      inst✝ : DivisionMonoid α
      a : α
      m n : Nat
      ⊢ Eq (HPow.hPow a (HMul.hMul ↑m.succ ↑n.succ)) (HPow.hPow a ↑(HMul.hMul (HAdd. …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive mul_zsmul]
                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : DivisionMonoid α
                                                                      a : α
                                                                      m n : Int
                                                                      ⊢ Eq (HPow.hPow a (HMul.hMul m n)) (HPow.hPow (HPow.hPow a n) m)
                                                                    -/
lemma zpow_mul' (a : α) (m n : ℤ) : a ^ (m * n) = (a ^ n) ^ m := by rw [Int.mul_comm, zpow_mul]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive, field_simps] -- The attributes are out of order on purpose
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DivisionMonoid α
                                                             a b c : α
                                                             ⊢ Eq (HDiv.hDiv a (HDiv.hDiv b c)) (HDiv.hDiv (HMul.hMul a c) b)
                                                           -/
theorem div_div_eq_mul_div : a / (b / c) = a * c / b := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DivisionMonoid α
                                                 a b : α
                                                 ⊢ Eq (HDiv.hDiv a (Inv.inv b)) (HMul.hMul a b)
                                               -/
theorem div_inv_eq_mul : a / b⁻¹ = a * b := by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem div_mul_eq_div_div_swap : a / (b * c) = a / c / b := by
  /-
    α : Type u_1
    inst✝ : DivisionMonoid α
    a b c : α
    ⊢ Eq (HDiv.hDiv a (HMul.hMul b c)) (HDiv.hDiv (HDiv.hDiv a c) b)
  -/
  simp only [mul_assoc, mul_inv_rev, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive neg_add]
                                              /-
                                                α : Type u_1
                                                inst✝ : DivisionCommMonoid α
                                                a b : α
                                                ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (Inv.inv b))
                                              -/
theorem mul_inv : (a * b)⁻¹ = a⁻¹ * b⁻¹ := by simp
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DivisionCommMonoid α
                                                 a b : α
                                                 ⊢ Eq (Inv.inv (HDiv.hDiv a b)) (HDiv.hDiv (Inv.inv a) (Inv.inv b))
                                               -/
theorem inv_div' : (a / b)⁻¹ = a⁻¹ / b⁻¹ := by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DivisionCommMonoid α
                                                 a b : α
                                                 ⊢ Eq (HDiv.hDiv a b) (HMul.hMul (Inv.inv b) a)
                                               -/
theorem div_eq_inv_mul : a / b = b⁻¹ * a := by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DivisionCommMonoid α
                                                 a b : α
                                                 ⊢ Eq (HMul.hMul (Inv.inv a) b) (HDiv.hDiv b a)
                                               -/
theorem inv_mul_eq_div : a⁻¹ * b = b / a := by simp
                                               /-
                                                 🎉 no goals
                                               -/


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : DivisionCommMonoid α
                                                                        a b : α
                                                                        ⊢ Eq (HDiv.hDiv (Inv.inv a) b) (HDiv.hDiv (Inv.inv b) a)
                                                                      -/
@[to_additive] lemma inv_div_comm (a b : α) : a⁻¹ / b = b⁻¹ / a := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive]
                                             /-
                                               α : Type u_1
                                               inst✝ : DivisionCommMonoid α
                                               a b : α
                                               ⊢ Eq (Inv.inv (HMul.hMul a b)) (HDiv.hDiv (Inv.inv a) b)
                                             -/
theorem inv_mul' : (a * b)⁻¹ = a⁻¹ / b := by simp
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive]
                                              /-
                                                α : Type u_1
                                                inst✝ : DivisionCommMonoid α
                                                a b : α
                                                ⊢ Eq (HDiv.hDiv (Inv.inv a) (Inv.inv b)) (HDiv.hDiv b a)
                                              -/
theorem inv_div_inv : a⁻¹ / b⁻¹ = b / a := by simp
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
                                                      /-
                                                        α : Type u_1
                                                        inst✝ : DivisionCommMonoid α
                                                        a b : α
                                                        ⊢ Eq (Inv.inv (HDiv.hDiv (Inv.inv a) (Inv.inv b))) (HDiv.hDiv a b)
                                                      -/
theorem inv_inv_div_inv : (a⁻¹ / b⁻¹)⁻¹ = a / b := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : DivisionCommMonoid α
                                                                    a b : α
                                                                    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) (HDiv.hDiv 1 b)) (HDiv.hDiv 1 (HMul.hMul a b))
                                                                  -/
theorem one_div_mul_one_div : 1 / a * (1 / b) = 1 / (a * b) := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : DivisionCommMonoid α
                                                       a b c : α
                                                       ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) c) (HDiv.hDiv (HDiv.hDiv a c) b)
                                                     -/
theorem div_right_comm : a / b / c = a / c / b := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive, field_simps]
                                                /-
                                                  α : Type u_1
                                                  inst✝ : DivisionCommMonoid α
                                                  a b c : α
                                                  ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) c) (HDiv.hDiv a (HMul.hMul b c))
                                                -/
theorem div_div : a / b / c = a / (b * c) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
                                                /-
                                                  α : Type u_1
                                                  inst✝ : DivisionCommMonoid α
                                                  a b c : α
                                                  ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HDiv.hDiv a (HDiv.hDiv b c))
                                                -/
theorem div_mul : a / b * c = a / (b / c) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
                                                            /-
                                                              α : Type u_1
                                                              inst✝ : DivisionCommMonoid α
                                                              a b c : α
                                                              ⊢ Eq (HMul.hMul a (HDiv.hDiv b c)) (HMul.hMul b (HDiv.hDiv a c))
                                                            -/
theorem mul_div_left_comm : a * (b / c) = b * (a / c) := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
                                                         /-
                                                           α : Type u_1
                                                           inst✝ : DivisionCommMonoid α
                                                           a b c : α
                                                           ⊢ Eq (HDiv.hDiv (HMul.hMul a b) c) (HMul.hMul (HDiv.hDiv a c) b)
                                                         -/
theorem mul_div_right_comm : a * b / c = a / c * b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DivisionCommMonoid α
                                                             a b c : α
                                                             ⊢ Eq (HDiv.hDiv a (HMul.hMul b c)) (HDiv.hDiv (HDiv.hDiv a b) c)
                                                           -/
theorem div_mul_eq_div_div : a / (b * c) = a / b / c := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive, field_simps]
                                                         /-
                                                           α : Type u_1
                                                           inst✝ : DivisionCommMonoid α
                                                           a b c : α
                                                           ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HDiv.hDiv (HMul.hMul a c) b)
                                                         -/
theorem div_mul_eq_mul_div : a / b * c = a * c / b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : DivisionCommMonoid α
                                                       a b : α
                                                       ⊢ Eq (HMul.hMul (HDiv.hDiv 1 a) b) (HDiv.hDiv b a)
                                                     -/
theorem one_div_mul_eq_div : 1 / a * b = b / a := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : DivisionCommMonoid α
                                                       a b c : α
                                                       ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HMul.hMul a (HDiv.hDiv c b))
                                                     -/
theorem mul_comm_div : a / b * c = a * (c / b) := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : DivisionCommMonoid α
                                                     a b c : α
                                                     ⊢ Eq (HMul.hMul (HDiv.hDiv a b) c) (HMul.hMul (HDiv.hDiv c b) a)
                                                   -/
theorem div_mul_comm : a / b * c = c / b * a := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
                                                                         /-
                                                                           α : Type u_1
                                                                           inst✝ : DivisionCommMonoid α
                                                                           a b c : α
                                                                           ⊢ Eq (HDiv.hDiv a (HMul.hMul b c)) (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv 1 c))
                                                                         -/
theorem div_mul_eq_div_mul_one_div : a / (b * c) = a / b * (1 / c) := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[to_additive]
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : DivisionCommMonoid α
                                                                   a b c d : α
                                                                   ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) (HDiv.hDiv c d)) (HDiv.hDiv (HMul.hMul a d) (H …
                                                                 -/
theorem div_div_div_eq : a / b / (c / d) = a * d / (b * c) := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DivisionCommMonoid α
                                                                     a b c d : α
                                                                     ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) (HDiv.hDiv c d)) (HDiv.hDiv (HDiv.hDiv a c) (H …
                                                                   -/
theorem div_div_div_comm : a / b / (c / d) = a / c / (b / d) := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DivisionCommMonoid α
                                                                     a b c d : α
                                                                     ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv c d)) (HDiv.hDiv (HMul.hMul a c) (H …
                                                                   -/
theorem div_mul_div_comm : a / b * (c / d) = a * c / (b * d) := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : DivisionCommMonoid α
                                                                     a b c d : α
                                                                     ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HDiv.hDiv a c) (H …
                                                                   -/
theorem mul_div_mul_comm : a * b / (c * d) = a / c * (b / d) := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive zsmul_add] lemma mul_zpow : ∀ n : ℤ, (a * b) ^ n = a ^ n * b ^ n
                  /-
                    α : Type u_1
                    inst✝ : DivisionCommMonoid α
                    a b : α
                    n : Nat
                    ⊢ Eq (HPow.hPow (HMul.hMul a b) ↑n) (HMul.hMul (HPow.hPow a ↑n) (HPow.hPow b ↑ …
                  -/
  | (n : ℕ) => by simp_rw [zpow_natCast, mul_pow]
                  /-
                    🎉 no goals
                  -/
                     /-
                       α : Type u_1
                       inst✝ : DivisionCommMonoid α
                       a b : α
                       n : Nat
                       ⊢ Eq (HPow.hPow (HMul.hMul a b) (Int.negSucc n)) (HMul.hMul (HPow.hPow a (Int. …
                     -/
  | .negSucc n => by simp_rw [zpow_negSucc, ← inv_pow, mul_inv, mul_pow]
                     /-
                       🎉 no goals
                     -/


@[to_additive nsmul_sub]
lemma div_pow (a b : α) (n : ℕ) : (a / b) ^ n = a ^ n / b ^ n := by
  /-
    α : Type u_1
    inst✝ : DivisionCommMonoid α
    a b : α
    n : Nat
    ⊢ Eq (HPow.hPow (HDiv.hDiv a b) n) (HDiv.hDiv (HPow.hPow a n) (HPow.hPow b n))
  -/
  simp only [div_eq_mul_inv, mul_pow, inv_pow]
  /-
    🎉 no goals
  -/


@[to_additive zsmul_sub]
lemma div_zpow (a b : α) (n : ℤ) : (a / b) ^ n = a ^ n / b ^ n := by
  /-
    α : Type u_1
    inst✝ : DivisionCommMonoid α
    a b : α
    n : Int
    ⊢ Eq (HPow.hPow (HDiv.hDiv a b) n) (HDiv.hDiv (HPow.hPow a n) (HPow.hPow b n))
  -/
  simp only [div_eq_mul_inv, mul_zpow, inv_zpow]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                    /-
                                                      G : Type u_3
                                                      inst✝ : Group G
                                                      a b : G
                                                      ⊢ Iff (Eq (HDiv.hDiv a b) (Inv.inv b)) (Eq a 1)
                                                    -/
theorem div_eq_inv_self : a / b = b⁻¹ ↔ a = 1 := by rw [div_eq_mul_inv, mul_left_eq_self]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive]
theorem mul_left_surjective (a : G) : Surjective (a * ·) :=
  fun x ↦ ⟨a⁻¹ * x, mul_inv_cancel_left a x⟩


@[to_additive]
theorem mul_right_surjective (a : G) : Function.Surjective fun x ↦ x * a := fun x ↦
  ⟨x * a⁻¹, inv_mul_cancel_right x a⟩


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq (HMul.hMul a c) b
                                                                   ⊢ Eq a (HMul.hMul b (Inv.inv c))
                                                                 -/
theorem eq_mul_inv_of_mul_eq (h : a * c = b) : a = b * c⁻¹ := by simp [h.symm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq (HMul.hMul b a) c
                                                                   ⊢ Eq a (HMul.hMul (Inv.inv b) c)
                                                                 -/
theorem eq_inv_mul_of_mul_eq (h : b * a = c) : a = b⁻¹ * c := by simp [h.symm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq b (HMul.hMul a c)
                                                                   ⊢ Eq (HMul.hMul (Inv.inv a) b) c
                                                                 -/
theorem inv_mul_eq_of_eq_mul (h : b = a * c) : a⁻¹ * b = c := by simp [h]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq a (HMul.hMul c b)
                                                                   ⊢ Eq (HMul.hMul a (Inv.inv b)) c
                                                                 -/
theorem mul_inv_eq_of_eq_mul (h : a = c * b) : a * b⁻¹ = c := by simp [h]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq (HMul.hMul a (Inv.inv c)) b
                                                                   ⊢ Eq a (HMul.hMul b c)
                                                                 -/
theorem eq_mul_of_mul_inv_eq (h : a * c⁻¹ = b) : a = b * c := by simp [h.symm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq (HMul.hMul (Inv.inv b) a) c
                                                                   ⊢ Eq a (HMul.hMul b c)
                                                                 -/
theorem eq_mul_of_inv_mul_eq (h : b⁻¹ * a = c) : a = b * c := by simp [h.symm, mul_inv_cancel_left]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq b (HMul.hMul (Inv.inv a) c)
                                                                   ⊢ Eq (HMul.hMul a b) c
                                                                 -/
theorem mul_eq_of_eq_inv_mul (h : b = a⁻¹ * c) : a * b = c := by rw [h, mul_inv_cancel_left]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
                                                                 /-
                                                                   G : Type u_3
                                                                   inst✝ : Group G
                                                                   a b c : G
                                                                   h : Eq a (HMul.hMul c (Inv.inv b))
                                                                   ⊢ Eq (HMul.hMul a b) c
                                                                 -/
theorem mul_eq_of_eq_mul_inv (h : a = c * b⁻¹) : a * b = c := by simp [h]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem mul_eq_one_iff_eq_inv : a * b = 1 ↔ a = b⁻¹ :=
                                         /-
                                           G : Type u_3
                                           inst✝ : Group G
                                           a b : G
                                           h : Eq a (Inv.inv b)
                                           ⊢ Eq (HMul.hMul a b) 1
                                         -/
  ⟨eq_inv_of_mul_eq_one_left, fun h ↦ by rw [h, inv_mul_cancel]⟩
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive]
theorem mul_eq_one_iff_inv_eq : a * b = 1 ↔ a⁻¹ = b := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Iff (Eq (HMul.hMul a b) 1) (Eq (Inv.inv a) b)
  -/
  rw [mul_eq_one_iff_eq_inv, inv_eq_iff_eq_inv]
  /-
    🎉 no goals
  -/


/-- Variant of `mul_eq_one_iff_eq_inv` with swapped equality. -/
@[to_additive]
theorem mul_eq_one_iff_eq_inv' : a * b = 1 ↔ b = a⁻¹ := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Iff (Eq (HMul.hMul a b) 1) (Eq b (Inv.inv a))
  -/
  rw [mul_eq_one_iff_inv_eq, eq_comm]
  /-
    🎉 no goals
  -/


/-- Variant of `mul_eq_one_iff_inv_eq` with swapped equality. -/
@[to_additive]
theorem mul_eq_one_iff_inv_eq' : a * b = 1 ↔ b⁻¹ = a := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Iff (Eq (HMul.hMul a b) 1) (Eq (Inv.inv b) a)
  -/
  rw [mul_eq_one_iff_eq_inv, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_inv_iff_mul_eq_one : a = b⁻¹ ↔ a * b = 1 :=
  mul_eq_one_iff_eq_inv.symm


@[to_additive]
theorem inv_eq_iff_mul_eq_one : a⁻¹ = b ↔ a * b = 1 :=
  mul_eq_one_iff_inv_eq.symm


@[to_additive]
theorem eq_mul_inv_iff_mul_eq : a = b * c⁻¹ ↔ a * c = b :=
              /-
                G : Type u_3
                inst✝ : Group G
                a b c : G
                h : Eq a (HMul.hMul b (Inv.inv c))
                ⊢ Eq (HMul.hMul a c) b
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [h, inv_mul_cancel_right], fun h ↦ by rw [← h, mul_inv_cancel_right]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive]
theorem eq_inv_mul_iff_mul_eq : a = b⁻¹ * c ↔ b * a = c :=
              /-
                G : Type u_3
                inst✝ : Group G
                a b c : G
                h : Eq a (HMul.hMul (Inv.inv b) c)
                ⊢ Eq (HMul.hMul b a) c
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [h, mul_inv_cancel_left], fun h ↦ by rw [← h, inv_mul_cancel_left]⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
theorem inv_mul_eq_iff_eq_mul : a⁻¹ * b = c ↔ b = a * c :=
              /-
                G : Type u_3
                inst✝ : Group G
                a b c : G
                h : Eq (HMul.hMul (Inv.inv a) b) c
                ⊢ Eq b (HMul.hMul a c)
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [← h, mul_inv_cancel_left], fun h ↦ by rw [h, inv_mul_cancel_left]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
theorem mul_inv_eq_iff_eq_mul : a * b⁻¹ = c ↔ a = c * b :=
              /-
                G : Type u_3
                inst✝ : Group G
                a b c : G
                h : Eq (HMul.hMul a (Inv.inv b)) c
                ⊢ Eq a (HMul.hMul c b)
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [← h, inv_mul_cancel_right], fun h ↦ by rw [h, mul_inv_cancel_right]⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
                                                   /-
                                                     G : Type u_3
                                                     inst✝ : Group G
                                                     a b : G
                                                     ⊢ Iff (Eq (HMul.hMul a (Inv.inv b)) 1) (Eq a b)
                                                   -/
theorem mul_inv_eq_one : a * b⁻¹ = 1 ↔ a = b := by rw [mul_eq_one_iff_eq_inv, inv_inv]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
                                                   /-
                                                     G : Type u_3
                                                     inst✝ : Group G
                                                     a b : G
                                                     ⊢ Iff (Eq (HMul.hMul (Inv.inv a) b) 1) (Eq a b)
                                                   -/
theorem inv_mul_eq_one : a⁻¹ * b = 1 ↔ a = b := by rw [mul_eq_one_iff_eq_inv, inv_inj]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive (attr := simp)]
theorem conj_eq_one_iff : a * b * a⁻¹ = 1 ↔ b = 1 := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul a b) (Inv.inv a)) 1) (Eq b 1)
  -/
  rw [mul_inv_eq_one, mul_right_eq_self]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem div_left_injective : Function.Injective fun a ↦ a / b := by
  -- FIXME this could be by `simpa`, but it fails. This is probably a bug in `simpa`.
  /-
    G : Type u_3
    inst✝ : Group G
    b : G
    ⊢ Function.Injective fun a => HDiv.hDiv a b
  -/
  simp only [div_eq_mul_inv]
  /-
    G : Type u_3
    inst✝ : Group G
    b : G
    ⊢ Function.Injective fun a => HMul.hMul a (Inv.inv b)
  -/
  exact fun a a' h ↦ mul_left_injective b⁻¹ h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem div_right_injective : Function.Injective fun a ↦ b / a := by
  -- FIXME see above
  /-
    G : Type u_3
    inst✝ : Group G
    b : G
    ⊢ Function.Injective fun a => HDiv.hDiv b a
  -/
  simp only [div_eq_mul_inv]
  /-
    G : Type u_3
    inst✝ : Group G
    b : G
    ⊢ Function.Injective fun a => HMul.hMul b (Inv.inv a)
  -/
  exact fun a a' h ↦ inv_injective (mul_right_injective b h)
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_mul_cancel (a b : G) : a / b * b = a := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) b) a
  -/
  rw [div_eq_mul_inv, inv_mul_cancel_right a b]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) sub_self]
                                            /-
                                              G : Type u_3
                                              inst✝ : Group G
                                              a : G
                                              ⊢ Eq (HDiv.hDiv a a) 1
                                            -/
theorem div_self' (a : G) : a / a = 1 := by rw [div_eq_mul_inv, mul_inv_cancel a]
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive (attr := simp)]
theorem mul_div_cancel_right (a b : G) : a * b / b = a := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b : G
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) b) a
  -/
  rw [div_eq_mul_inv, mul_inv_cancel_right a b]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                               /-
                                                                 G : Type u_3
                                                                 inst✝ : Group G
                                                                 a b : G
                                                                 ⊢ Eq (HDiv.hDiv a (HMul.hMul b a)) (Inv.inv b)
                                                               -/
lemma div_mul_cancel_right (a b : G) : a / (b * a) = b⁻¹ := by rw [← inv_div, mul_div_cancel_right]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive (attr := simp)]
theorem mul_div_mul_right_eq_div (a b c : G) : a * c / (b * c) = a / b := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b c : G
    ⊢ Eq (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c)) (HDiv.hDiv a b)
  -/
  rw [div_mul_eq_div_div_swap]; simp only [mul_left_inj, eq_self_iff_true, mul_div_cancel_right]
                                /-
                                  🎉 no goals
                                -/


@[to_additive eq_sub_of_add_eq]
                                                            /-
                                                              G : Type u_3
                                                              inst✝ : Group G
                                                              a b c : G
                                                              h : Eq (HMul.hMul a c) b
                                                              ⊢ Eq a (HDiv.hDiv b c)
                                                            -/
theorem eq_div_of_mul_eq' (h : a * c = b) : a = b / c := by simp [← h]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive sub_eq_of_eq_add]
                                                             /-
                                                               G : Type u_3
                                                               inst✝ : Group G
                                                               a b c : G
                                                               h : Eq a (HMul.hMul c b)
                                                               ⊢ Eq (HDiv.hDiv a b) c
                                                             -/
theorem div_eq_of_eq_mul'' (h : a = c * b) : a / b = c := by simp [h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
                                                           /-
                                                             G : Type u_3
                                                             inst✝ : Group G
                                                             a b c : G
                                                             h : Eq (HDiv.hDiv a c) b
                                                             ⊢ Eq a (HMul.hMul b c)
                                                           -/
theorem eq_mul_of_div_eq (h : a / c = b) : a = b * c := by simp [← h]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
                                                           /-
                                                             G : Type u_3
                                                             inst✝ : Group G
                                                             a b c : G
                                                             h : Eq a (HDiv.hDiv c b)
                                                             ⊢ Eq (HMul.hMul a b) c
                                                           -/
theorem mul_eq_of_eq_div (h : a = c / b) : a * b = c := by simp [h]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
theorem div_right_inj : a / b = a / c ↔ b = c :=
  div_right_injective.eq_iff


@[to_additive (attr := simp)]
theorem div_left_inj : b / a = c / a ↔ b = c := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b c : G
    ⊢ Iff (Eq (HDiv.hDiv b a) (HDiv.hDiv c a)) (Eq b c)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G : Type u_3
    inst✝ : Group G
    a b c : G
    ⊢ Iff (Eq (HMul.hMul b (Inv.inv a)) (HMul.hMul c (Inv.inv a))) (Eq b c)
  -/
  exact mul_left_inj _
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_mul_div_cancel (a b c : G) : a / b * (b / c) = a / c := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b c : G
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv b c)) (HDiv.hDiv a c)
  -/
  rw [← mul_div_assoc, div_mul_cancel]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_div_div_cancel_right (a b c : G) : a / c / (b / c) = a / b := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b c : G
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv a b)
  -/
  rw [← inv_div c b, div_inv_eq_mul, div_mul_div_cancel]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-24")] alias div_div_div_cancel_right' := div_div_div_cancel_right


@[to_additive]
theorem div_eq_one : a / b = 1 ↔ a = b :=
                                /-
                                  G : Type u_3
                                  inst✝ : Group G
                                  a b : G
                                  h : Eq a b
                                  ⊢ Eq (HDiv.hDiv a b) 1
                                -/
  ⟨eq_of_div_eq_one, fun h ↦ by rw [h, div_self']⟩
                                /-
                                  🎉 no goals
                                -/


alias ⟨_, div_eq_one_of_eq⟩ := div_eq_one


alias ⟨_, sub_eq_zero_of_eq⟩ := sub_eq_zero


@[to_additive]
theorem div_ne_one : a / b ≠ 1 ↔ a ≠ b :=
  not_congr div_eq_one


@[to_additive (attr := simp)]
                                              /-
                                                G : Type u_3
                                                inst✝ : Group G
                                                a b : G
                                                ⊢ Iff (Eq (HDiv.hDiv a b) a) (Eq b 1)
                                              -/
theorem div_eq_self : a / b = a ↔ b = 1 := by rw [div_eq_mul_inv, mul_right_eq_self, inv_eq_one]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive eq_sub_iff_add_eq]
                                                         /-
                                                           G : Type u_3
                                                           inst✝ : Group G
                                                           a b c : G
                                                           ⊢ Iff (Eq a (HDiv.hDiv b c)) (Eq (HMul.hMul a c) b)
                                                         -/
theorem eq_div_iff_mul_eq' : a = b / c ↔ a * c = b := by rw [div_eq_mul_inv, eq_mul_inv_iff_mul_eq]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
                                                        /-
                                                          G : Type u_3
                                                          inst✝ : Group G
                                                          a b c : G
                                                          ⊢ Iff (Eq (HDiv.hDiv a b) c) (Eq a (HMul.hMul c b))
                                                        -/
theorem div_eq_iff_eq_mul : a / b = c ↔ a = c * b := by rw [div_eq_mul_inv, mul_inv_eq_iff_eq_mul]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
theorem eq_iff_eq_of_div_eq_div (H : a / b = c / d) : a = b ↔ c = d := by
  /-
    G : Type u_3
    inst✝ : Group G
    a b c d : G
    H : Eq (HDiv.hDiv a b) (HDiv.hDiv c d)
    ⊢ Iff (Eq a b) (Eq c d)
  -/
  rw [← div_eq_one, H, div_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem leftInverse_div_mul_left (c : G) : Function.LeftInverse (fun x ↦ x / c) fun x ↦ x * c :=
  fun x ↦ mul_div_cancel_right x c


@[to_additive]
theorem leftInverse_mul_left_div (c : G) : Function.LeftInverse (fun x ↦ x * c) fun x ↦ x / c :=
  fun x ↦ div_mul_cancel x c


@[to_additive]
theorem leftInverse_mul_right_inv_mul (c : G) :
    Function.LeftInverse (fun x ↦ c * x) fun x ↦ c⁻¹ * x :=
  fun x ↦ mul_inv_cancel_left c x


@[to_additive]
theorem leftInverse_inv_mul_mul_right (c : G) :
    Function.LeftInverse (fun x ↦ c⁻¹ * x) fun x ↦ c * x :=
  fun x ↦ inv_mul_cancel_left c x


@[to_additive (attr := simp) natAbs_nsmul_eq_zero]
                                                             /-
                                                               G : Type u_3
                                                               inst✝ : Group G
                                                               a : G
                                                               n : Int
                                                               ⊢ Iff (Eq (HPow.hPow a n.natAbs) 1) (Eq (HPow.hPow a n) 1)
                                                             -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
lemma pow_natAbs_eq_one : a ^ n.natAbs = 1 ↔ a ^ n = 1 := by cases n <;> simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


set_option linter.existingAttributeWarning false in
@[to_additive, deprecated pow_natAbs_eq_one (since := "2024-02-14")]
lemma exists_pow_eq_one_of_zpow_eq_one (hn : n ≠ 0) (h : a ^ n = 1) :
    ∃ n : ℕ, 0 < n ∧ a ^ n = 1 := ⟨_, Int.natAbs_pos.2 hn, pow_natAbs_eq_one.2 h⟩


@[to_additive sub_nsmul]
lemma pow_sub (a : G) {m n : ℕ} (h : n ≤ m) : a ^ (m - n) = a ^ m * (a ^ n)⁻¹ :=
                             /-
                               G : Type u_3
                               inst✝ : Group G
                               a : G
                               m n : Nat
                               h : LE.le n m
                               ⊢ Eq (HMul.hMul (HPow.hPow a (HSub.hSub m n)) (HPow.hPow a n)) (HPow.hPow a m)
                             -/
  eq_mul_inv_of_mul_eq <| by rw [← pow_add, Nat.sub_add_cancel h]
                             /-
                               🎉 no goals
                             -/


@[to_additive sub_nsmul_neg]
theorem inv_pow_sub (a : G) {m n : ℕ} (h : n ≤ m) : a⁻¹ ^ (m - n) = (a ^ m)⁻¹ * a ^ n := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    m n : Nat
    h : LE.le n m
    ⊢ Eq (HPow.hPow (Inv.inv a) (HSub.hSub m n)) (HMul.hMul (Inv.inv (HPow.hPow a  …
  -/
  rw [pow_sub a⁻¹ h, inv_pow, inv_pow, inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive add_one_zsmul]
lemma zpow_add_one (a : G) : ∀ n : ℤ, a ^ (n + 1) = a ^ n * a
                  /-
                    G : Type u_3
                    inst✝ : Group G
                    a : G
                    n : Nat
                    ⊢ Eq (HPow.hPow a (HAdd.hAdd (↑n) 1)) (HMul.hMul (HPow.hPow a ↑n) a)
                  -/
  | (n : ℕ) => by simp only [← Int.ofNat_succ, zpow_natCast, pow_succ]
                  /-
                    🎉 no goals
                  -/
                     /-
                       G : Type u_3
                       inst✝ : Group G
                       a : G
                       ⊢ Eq (HPow.hPow a (HAdd.hAdd (Int.negSucc 0) 1)) (HMul.hMul (HPow.hPow a (Int. …
                     -/
  | .negSucc 0 => by simp [Int.negSucc_eq', Int.add_left_neg]
                     /-
                       🎉 no goals
                     -/
  | .negSucc (n + 1) => by
    /-
      G : Type u_3
      inst✝ : Group G
      a : G
      n : Nat
      ⊢ Eq (HPow.hPow a (HAdd.hAdd (Int.negSucc (HAdd.hAdd n 1)) 1)) (HMul.hMul (HPo …
    -/
    rw [zpow_negSucc, pow_succ', mul_inv_rev, inv_mul_cancel_right]
    /-
      G : Type u_3
      inst✝ : Group G
      a : G
      n : Nat
      ⊢ Eq (HPow.hPow a (HAdd.hAdd (Int.negSucc (HAdd.hAdd n 1)) 1)) (Inv.inv (HPow. …
    -/
    rw [Int.negSucc_eq, Int.neg_add, Int.neg_add_cancel_right]
    /-
      G : Type u_3
      inst✝ : Group G
      a : G
      n : Nat
      ⊢ Eq (HPow.hPow a (Neg.neg ↑(HAdd.hAdd n 1))) (Inv.inv (HPow.hPow a (HAdd.hAdd …
    -/
    exact zpow_negSucc _ _
    /-
      🎉 no goals
    -/


@[to_additive sub_one_zsmul]
lemma zpow_sub_one (a : G) (n : ℤ) : a ^ (n - 1) = a ^ n * a⁻¹ :=
  calc
    a ^ (n - 1) = a ^ (n - 1) * a * a⁻¹ := (mul_inv_cancel_right _ _).symm
                          /-
                            G : Type u_3
                            inst✝ : Group G
                            a : G
                            n : Int
                            ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a (HSub.hSub n 1)) a) (Inv.inv a)) (HMul …
                          -/
    _ = a ^ n * a⁻¹ := by rw [← zpow_add_one, Int.sub_add_cancel]
                          /-
                            🎉 no goals
                          -/


@[to_additive add_zsmul]
lemma zpow_add (a : G) (m n : ℤ) : a ^ (m + n) = a ^ m * a ^ n := by
  induction n using Int.induction_on with
  | hz => simp
  | hp n ihn => simp only [← Int.add_assoc, zpow_add_one, ihn, mul_assoc]
  | hn n ihn => rw [zpow_sub_one, ← mul_assoc, ← ihn, ← zpow_sub_one, Int.add_sub_assoc]


@[to_additive one_add_zsmul]
                                                                   /-
                                                                     G : Type u_3
                                                                     inst✝ : Group G
                                                                     a : G
                                                                     n : Int
                                                                     ⊢ Eq (HPow.hPow a (HAdd.hAdd 1 n)) (HMul.hMul a (HPow.hPow a n))
                                                                   -/
lemma zpow_one_add (a : G) (n : ℤ) : a ^ (1 + n) = a * a ^ n := by rw [zpow_add, zpow_one]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive add_zsmul_self]
lemma mul_self_zpow (a : G) (n : ℤ) : a * a ^ n = a ^ (n + 1) := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    n : Int
    ⊢ Eq (HMul.hMul a (HPow.hPow a n)) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  rw [Int.add_comm, zpow_add, zpow_one]
  /-
    🎉 no goals
  -/


@[to_additive add_self_zsmul]
lemma mul_zpow_self (a : G) (n : ℤ) : a ^ n * a = a ^ (n + 1) := (zpow_add_one ..).symm


@[to_additive sub_zsmul] lemma zpow_sub (a : G) (m n : ℤ) : a ^ (m - n) = a ^ m * (a ^ n)⁻¹ := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    m n : Int
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
  -/
  rw [Int.sub_eq_add_neg, zpow_add, zpow_neg]
  /-
    🎉 no goals
  -/


@[to_additive natCast_sub_natCast_zsmul]
lemma zpow_natCast_sub_natCast (a : G) (m n : ℕ) : a ^ (m - n : ℤ) = a ^ m / a ^ n := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    m n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub ↑m ↑n)) (HDiv.hDiv (HPow.hPow a m) (HPow.hPow a n))
  -/
  simpa [div_eq_mul_inv] using zpow_sub a m n
  /-
    🎉 no goals
  -/


@[to_additive natCast_sub_one_zsmul]
lemma zpow_natCast_sub_one (a : G) (n : ℕ) : a ^ (n - 1 : ℤ) = a ^ n / a := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub (↑n) 1)) (HDiv.hDiv (HPow.hPow a n) a)
  -/
  simpa [div_eq_mul_inv] using zpow_sub a n 1
  /-
    🎉 no goals
  -/


@[to_additive one_sub_natCast_zsmul]
lemma zpow_one_sub_natCast (a : G) (n : ℕ) : a ^ (1 - n : ℤ) = a / a ^ n := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub 1 ↑n)) (HDiv.hDiv a (HPow.hPow a n))
  -/
  simpa [div_eq_mul_inv] using zpow_sub a 1 n
  /-
    🎉 no goals
  -/


@[to_additive] lemma zpow_mul_comm (a : G) (m n : ℤ) : a ^ m * a ^ n = a ^ n * a ^ m := by
  /-
    G : Type u_3
    inst✝ : Group G
    a : G
    m n : Int
    ⊢ Eq (HMul.hMul (HPow.hPow a m) (HPow.hPow a n)) (HMul.hMul (HPow.hPow a n) (H …
  -/
  rw [← zpow_add, Int.add_comm, zpow_add]
  /-
    🎉 no goals
  -/


theorem zpow_eq_zpow_emod {x : G} (m : ℤ) {n : ℤ} (h : x ^ n = 1) :
    x ^ m = x ^ (m % n) :=
  calc
                                            /-
                                              G : Type u_3
                                              inst✝ : Group G
                                              x : G
                                              m n : Int
                                              h : Eq (HPow.hPow x n) 1
                                              ⊢ Eq (HPow.hPow x m) (HPow.hPow x (HAdd.hAdd (HMod.hMod m n) (HMul.hMul n (HDi …
                                            -/
    x ^ m = x ^ (m % n + n * (m / n)) := by rw [Int.emod_add_ediv]
                                            /-
                                              🎉 no goals
                                            -/
                          /-
                            G : Type u_3
                            inst✝ : Group G
                            x : G
                            m n : Int
                            h : Eq (HPow.hPow x n) 1
                            ⊢ Eq (HPow.hPow x (HAdd.hAdd (HMod.hMod m n) (HMul.hMul n (HDiv.hDiv m n)))) ( …
                          -/
    _ = x ^ (m % n) := by simp [zpow_add, zpow_mul, h]
                          /-
                            🎉 no goals
                          -/


theorem zpow_eq_zpow_emod' {x : G} (m : ℤ) {n : ℕ} (h : x ^ n = 1) :
                                                         /-
                                                           G : Type u_3
                                                           inst✝ : Group G
                                                           x : G
                                                           m : Int
                                                           n : Nat
                                                           h : Eq (HPow.hPow x n) 1
                                                           ⊢ Eq (HPow.hPow x ↑n) 1
                                                         -/
    x ^ m = x ^ (m % (n : ℤ)) := zpow_eq_zpow_emod m (by simpa)
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- To show a property of all powers of `g` it suffices to show it is closed under multiplication
by `g` and `g⁻¹` on the left. For subgroups generated by more than one element, see
`Subgroup.closure_induction_left`. -/
@[to_additive "To show a property of all multiples of `g` it suffices to show it is closed under
addition by `g` and `-g` on the left. For additive subgroups generated by more than one element, see
`AddSubgroup.closure_induction_left`."]
lemma zpow_induction_left {g : G} {P : G → Prop} (h_one : P (1 : G))
    (h_mul : ∀ a, P a → P (g * a)) (h_inv : ∀ a, P a → P (g⁻¹ * a)) (n : ℤ) : P (g ^ n) := by
  induction n using Int.induction_on with
  | hz => rwa [zpow_zero]
  | hp n ih =>
    rw [Int.add_comm, zpow_add, zpow_one]
    exact h_mul _ ih
  | hn n ih =>
    rw [Int.sub_eq_add_neg, Int.add_comm, zpow_add, zpow_neg_one]
    exact h_inv _ ih


/-- To show a property of all powers of `g` it suffices to show it is closed under multiplication
by `g` and `g⁻¹` on the right. For subgroups generated by more than one element, see
`Subgroup.closure_induction_right`. -/
@[to_additive "To show a property of all multiples of `g` it suffices to show it is closed under
addition by `g` and `-g` on the right. For additive subgroups generated by more than one element,
see `AddSubgroup.closure_induction_right`."]
lemma zpow_induction_right {g : G} {P : G → Prop} (h_one : P (1 : G))
    (h_mul : ∀ a, P a → P (a * g)) (h_inv : ∀ a, P a → P (a * g⁻¹)) (n : ℤ) : P (g ^ n) := by
  induction n using Int.induction_on with
  | hz => rwa [zpow_zero]
  | hp n ih =>
    rw [zpow_add_one]
    exact h_mul _ ih
  | hn n ih =>
    rw [zpow_sub_one]
    exact h_inv _ ih


@[to_additive]
theorem div_eq_of_eq_mul' {a b c : G} (h : a = b * c) : a / b = c := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    h : Eq a (HMul.hMul b c)
    ⊢ Eq (HDiv.hDiv a b) c
  -/
  rw [h, div_eq_mul_inv, mul_comm, inv_mul_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_div_mul_left_eq_div (a b c : G) : c * a / (c * b) = a / b := by
  rw [div_eq_mul_inv, mul_inv_rev, mul_comm b⁻¹ c⁻¹, mul_comm c a, mul_assoc, ← mul_assoc c,
    mul_inv_cancel, one_mul, div_eq_mul_inv]


@[to_additive eq_sub_of_add_eq']
                                                             /-
                                                               G : Type u_3
                                                               inst✝ : CommGroup G
                                                               a b c : G
                                                               h : Eq (HMul.hMul c a) b
                                                               ⊢ Eq a (HDiv.hDiv b c)
                                                             -/
theorem eq_div_of_mul_eq'' (h : c * a = b) : a = b / c := by simp [h.symm]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
                                                            /-
                                                              G : Type u_3
                                                              inst✝ : CommGroup G
                                                              a b c : G
                                                              h : Eq (HDiv.hDiv a b) c
                                                              ⊢ Eq a (HMul.hMul b c)
                                                            -/
theorem eq_mul_of_div_eq' (h : a / b = c) : a = b * c := by simp [h.symm]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
theorem mul_eq_of_eq_div' (h : b = c / a) : a * b = c := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    h : Eq b (HDiv.hDiv c a)
    ⊢ Eq (HMul.hMul a b) c
  -/
  rw [h, div_eq_mul_inv, mul_comm c, mul_inv_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive sub_sub_self]
                                                        /-
                                                          G : Type u_3
                                                          inst✝ : CommGroup G
                                                          a b : G
                                                          ⊢ Eq (HDiv.hDiv a (HDiv.hDiv a b)) b
                                                        -/
theorem div_div_self' (a b : G) : a / (a / b) = b := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
                                                                       /-
                                                                         G : Type u_3
                                                                         inst✝ : CommGroup G
                                                                         a b c : G
                                                                         ⊢ Eq (HDiv.hDiv a b) (HMul.hMul (HDiv.hDiv c b) (HDiv.hDiv a c))
                                                                       -/
theorem div_eq_div_mul_div (a b c : G) : a / b = c / b * (a / c) := by simp [mul_left_comm c]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive (attr := simp)]
theorem div_div_cancel (a b : G) : a / (a / b) = b :=
  div_div_self' a b


@[to_additive (attr := simp)]
                                                              /-
                                                                G : Type u_3
                                                                inst✝ : CommGroup G
                                                                a b : G
                                                                ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b) a) (Inv.inv b)
                                                              -/
theorem div_div_cancel_left (a b : G) : a / b / a = b⁻¹ := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive eq_sub_iff_add_eq']
                                                          /-
                                                            G : Type u_3
                                                            inst✝ : CommGroup G
                                                            a b c : G
                                                            ⊢ Iff (Eq a (HDiv.hDiv b c)) (Eq (HMul.hMul c a) b)
                                                          -/
theorem eq_div_iff_mul_eq'' : a = b / c ↔ c * a = b := by rw [eq_div_iff_mul_eq', mul_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive]
                                                         /-
                                                           G : Type u_3
                                                           inst✝ : CommGroup G
                                                           a b c : G
                                                           ⊢ Iff (Eq (HDiv.hDiv a b) c) (Eq a (HMul.hMul b c))
                                                         -/
theorem div_eq_iff_eq_mul' : a / b = c ↔ a = b * c := by rw [div_eq_iff_eq_mul, mul_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive (attr := simp)]
                                                            /-
                                                              G : Type u_3
                                                              inst✝ : CommGroup G
                                                              a b : G
                                                              ⊢ Eq (HDiv.hDiv (HMul.hMul a b) a) b
                                                            -/
theorem mul_div_cancel_left (a b : G) : a * b / a = b := by rw [div_eq_inv_mul, inv_mul_cancel_left]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive (attr := simp)]
theorem mul_div_cancel (a b : G) : a * (b / a) = b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b : G
    ⊢ Eq (HMul.hMul a (HDiv.hDiv b a)) b
  -/
  rw [← mul_div_assoc, mul_div_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                                /-
                                                                  G : Type u_3
                                                                  inst✝ : CommGroup G
                                                                  a b : G
                                                                  ⊢ Eq (HDiv.hDiv a (HMul.hMul a b)) (Inv.inv b)
                                                                -/
theorem div_mul_cancel_left (a b : G) : a / (a * b) = b⁻¹ := by rw [← inv_div, mul_div_cancel_left]
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- This lemma is in the `simp` set under the name `mul_inv_cancel_comm_assoc`,
-- along with the additive version `add_neg_cancel_comm_assoc`,
-- defined in `Algebra.Group.Commute`

@[to_additive]
theorem mul_mul_inv_cancel'_right (a b : G) : a * (b * a⁻¹) = b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b : G
    ⊢ Eq (HMul.hMul a (HMul.hMul b (Inv.inv a))) b
  -/
  rw [← div_eq_mul_inv, mul_div_cancel a b]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_mul_div_cancel (a b c : G) : a * c * (b / c) = a * b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    ⊢ Eq (HMul.hMul (HMul.hMul a c) (HDiv.hDiv b c)) (HMul.hMul a b)
  -/
  rw [mul_assoc, mul_div_cancel]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_mul_mul_cancel (a b c : G) : a / c * (b * c) = a * b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    ⊢ Eq (HMul.hMul (HDiv.hDiv a c) (HMul.hMul b c)) (HMul.hMul a b)
  -/
  rw [mul_left_comm, div_mul_cancel, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_mul_div_cancel' (a b c : G) : a / b * (c / a) = c / b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv c a)) (HDiv.hDiv c b)
  -/
  rw [mul_comm]; apply div_mul_div_cancel
                 /-
                   🎉 no goals
                 -/


@[deprecated (since := "2024-08-24")] alias div_mul_div_cancel'' := div_mul_div_cancel'


@[to_additive (attr := simp)]
theorem mul_div_div_cancel (a b c : G) : a * b / (a / c) = b * c := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HDiv.hDiv a c)) (HMul.hMul b c)
  -/
  rw [← div_mul, mul_div_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem div_div_div_cancel_left (a b c : G) : c / a / (c / b) = b / a := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c : G
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv c a) (HDiv.hDiv c b)) (HDiv.hDiv b a)
  -/
  rw [← inv_div b c, div_inv_eq_mul, mul_comm, div_mul_div_cancel]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem div_eq_div_iff_mul_eq_mul : a / b = c / d ↔ a * d = c * b := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c d : G
    ⊢ Iff (Eq (HDiv.hDiv a b) (HDiv.hDiv c d)) (Eq (HMul.hMul a d) (HMul.hMul c b))
  -/
  rw [div_eq_iff_eq_mul, div_mul_eq_mul_div, eq_comm, div_eq_iff_eq_mul']
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c d : G
    ⊢ Iff (Eq (HMul.hMul c b) (HMul.hMul d a)) (Eq (HMul.hMul a d) (HMul.hMul c b))
  -/
  simp only [mul_comm, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem div_eq_div_iff_div_eq_div : a / b = c / d ↔ a / c = b / d := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    a b c d : G
    ⊢ Iff (Eq (HDiv.hDiv a b) (HDiv.hDiv c d)) (Eq (HDiv.hDiv a c) (HDiv.hDiv b d))
  -/
  rw [div_eq_iff_eq_mul, div_mul_eq_mul_div, div_eq_iff_eq_mul', mul_div_assoc]
  /-
    🎉 no goals
  -/


@[to_additive additive_of_symmetric_of_isTotal]
lemma multiplicative_of_symmetric_of_isTotal
    (hsymm : Symmetric p) (hf_swap : ∀ {a b}, p a b → f a b * f b a = 1)
    (hmul : ∀ {a b c}, r a b → r b c → p a b → p b c → p a c → f a c = f a b * f b c)
    {a b c : α} (pab : p a b) (pbc : p b c) (pac : p a c) : f a c = f a b * f b c := by
  have hmul' : ∀ {b c}, r b c → p a b → p b c → p a c → f a c = f a b * f b c := by
    intros b c rbc pab pbc pac
    obtain rab | rba := total_of r a b
    · exact hmul rab rbc pab pbc pac
    rw [← one_mul (f a c), ← hf_swap pab, mul_assoc]
    obtain rac | rca := total_of r a c
    · rw [hmul rba rac (hsymm pab) pac pbc]
    · rw [hmul rbc rca pbc (hsymm pac) (hsymm pab), mul_assoc, hf_swap (hsymm pac), mul_one]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid β
    p r : α → α → Prop
    inst✝ : IsTotal α r
    f : α → α → β
    hsymm : Symmetric p
    hf_swap : ∀ {a b : α}, p a b → Eq (HMul.hMul (f a b) (f b a)) 1
    hmul : ∀ {a b c : α}, r a b → r b c → p a b → p b c → p a c → Eq (f a c) (HMul …
    a b c : α
    pab : p a b
    pbc : p b c
    pac : p a c
    hmul' : ∀ {b c : α}, r b c → p a b → p b c → p a c → Eq (f a c) (HMul.hMul (f  …
    ⊢ Eq (f a c) (HMul.hMul (f a b) (f b c))
  -/
  obtain rbc | rcb := total_of r b c
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : Monoid β
      p r : α → α → Prop
      inst✝ : IsTotal α r
      f : α → α → β
      hsymm : Symmetric p
      hf_swap : ∀ {a b : α}, p a b → Eq (HMul.hMul (f a b) (f b a)) 1
      hmul : ∀ {a b c : α}, r a b → r b c → p a b → p b c → p a c → Eq (f a c) (HMul …
      a b c : α
      pab : p a b
      pbc : p b c
      pac : p a c
      hmul' : ∀ {b c : α}, r b c → p a b → p b c → p a c → Eq (f a c) (HMul.hMul (f  …
      rbc : r b c
      ⊢ Eq (f a c) (HMul.hMul (f a b) (f b c))
    -/
  · exact hmul' rbc pab pbc pac
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : Monoid β
      p r : α → α → Prop
      inst✝ : IsTotal α r
      f : α → α → β
      hsymm : Symmetric p
      hf_swap : ∀ {a b : α}, p a b → Eq (HMul.hMul (f a b) (f b a)) 1
      hmul : ∀ {a b c : α}, r a b → r b c → p a b → p b c → p a c → Eq (f a c) (HMul …
      a b c : α
      pab : p a b
      pbc : p b c
      pac : p a c
      hmul' : ∀ {b c : α}, r b c → p a b → p b c → p a c → Eq (f a c) (HMul.hMul (f  …
      rcb : r c b
      ⊢ Eq (f a c) (HMul.hMul (f a b) (f b c))
    -/
  · rw [hmul' rcb pac (hsymm pbc) pab, mul_assoc, hf_swap (hsymm pbc), mul_one]
    /-
      🎉 no goals
    -/


/-- If a binary function from a type equipped with a total relation `r` to a monoid is
  anti-symmetric (i.e. satisfies `f a b * f b a = 1`), in order to show it is multiplicative
  (i.e. satisfies `f a c = f a b * f b c`), we may assume `r a b` and `r b c` are satisfied.
  We allow restricting to a subset specified by a predicate `p`. -/
@[to_additive additive_of_isTotal "If a binary function from a type equipped with a total relation
`r` to an additive monoid is anti-symmetric (i.e. satisfies `f a b + f b a = 0`), in order to show
it is additive (i.e. satisfies `f a c = f a b + f b c`), we may assume `r a b` and `r b c` are
satisfied. We allow restricting to a subset specified by a predicate `p`."]
theorem multiplicative_of_isTotal (p : α → Prop) (hswap : ∀ {a b}, p a → p b → f a b * f b a = 1)
    (hmul : ∀ {a b c}, r a b → r b c → p a → p b → p c → f a c = f a b * f b c) {a b c : α}
    (pa : p a) (pb : p b) (pc : p c) : f a c = f a b * f b c := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid β
    r : α → α → Prop
    inst✝ : IsTotal α r
    f : α → α → β
    p : α → Prop
    hswap : ∀ {a b : α}, p a → p b → Eq (HMul.hMul (f a b) (f b a)) 1
    hmul : ∀ {a b c : α}, r a b → r b c → p a → p b → p c → Eq (f a c) (HMul.hMul  …
    a b c : α
    pa : p a
    pb : p b
    pc : p c
    ⊢ Eq (f a c) (HMul.hMul (f a b) (f b c))
  -/
  apply multiplicative_of_symmetric_of_isTotal (fun a b => p a ∧ p b) r f fun _ _ => And.symm
    /-
      case hf_swap
      α : Type u_1
      β : Type u_2
      inst✝¹ : Monoid β
      r : α → α → Prop
      inst✝ : IsTotal α r
      f : α → α → β
      p : α → Prop
      hswap : ∀ {a b : α}, p a → p b → Eq (HMul.hMul (f a b) (f b a)) 1
      hmul : ∀ {a b c : α}, r a b → r b c → p a → p b → p c → Eq (f a c) (HMul.hMul  …
      a b c : α
      pa : p a
      pb : p b
      pc : p c
      ⊢ ∀ {a b : α}, And (p a) (p b) → Eq (HMul.hMul (f a b) (f b a)) 1
    -/
  · simp_rw [and_imp]; exact @hswap
                       /-
                         🎉 no goals
                       -/
    /-
      case hmul
      α : Type u_1
      β : Type u_2
      inst✝¹ : Monoid β
      r : α → α → Prop
      inst✝ : IsTotal α r
      f : α → α → β
      p : α → Prop
      hswap : ∀ {a b : α}, p a → p b → Eq (HMul.hMul (f a b) (f b a)) 1
      hmul : ∀ {a b c : α}, r a b → r b c → p a → p b → p c → Eq (f a c) (HMul.hMul  …
      a b c : α
      pa : p a
      pb : p b
      pc : p c
      ⊢ ∀ {a b c : α}, r a b → r b c → And (p a) (p b) → And (p b) (p c) → And (p a) …
    -/
  · exact fun rab rbc pab _pbc pac => hmul rab rbc pab.1 pab.2 pac.2
    /-
      🎉 no goals
    -/
  /-
    case pab
    α : Type u_1
    β : Type u_2
    inst✝¹ : Monoid β
    r : α → α → Prop
    inst✝ : IsTotal α r
    f : α → α → β
    p : α → Prop
    hswap : ∀ {a b : α}, p a → p b → Eq (HMul.hMul (f a b) (f b a)) 1
    hmul : ∀ {a b c : α}, r a b → r b c → p a → p b → p c → Eq (f a c) (HMul.hMul  …
    a b c : α
    pa : p a
    pb : p b
    pc : p c
    ⊢ And (p a) (p b)
  -/
  exacts [⟨pa, pb⟩, ⟨pb, pc⟩, ⟨pa, pc⟩]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-20")] alias div_mul_cancel' := div_mul_cancel

@[deprecated (since := "2024-03-20")] alias mul_div_cancel'' := mul_div_cancel_right
-- The name `add_sub_cancel` was reused
-- @[deprecated (since := "2024-03-20")] alias add_sub_cancel := add_sub_cancel_right

@[deprecated (since := "2024-03-20")] alias div_mul_cancel''' := div_mul_cancel_right

@[deprecated (since := "2024-03-20")] alias sub_add_cancel'' := sub_add_cancel_right

@[deprecated (since := "2024-03-20")] alias mul_div_cancel''' := mul_div_cancel_left

@[deprecated (since := "2024-03-20")] alias add_sub_cancel' := add_sub_cancel_left

@[deprecated (since := "2024-03-20")] alias mul_div_cancel'_right := mul_div_cancel

@[deprecated (since := "2024-03-20")] alias add_sub_cancel'_right := add_sub_cancel

@[deprecated (since := "2024-03-20")] alias div_mul_cancel'' := div_mul_cancel_left

@[deprecated (since := "2024-03-20")] alias sub_add_cancel' := sub_add_cancel_left

