theorem left_ne_zero_of_mul : a * b ≠ 0 → a ≠ 0 :=
  mt fun h => mul_eq_zero_of_left h b


theorem right_ne_zero_of_mul : a * b ≠ 0 → b ≠ 0 :=
  mt (mul_eq_zero_of_right a)


theorem ne_zero_and_ne_zero_of_mul (h : a * b ≠ 0) : a ≠ 0 ∧ b ≠ 0 :=
  ⟨left_ne_zero_of_mul h, right_ne_zero_of_mul h⟩


theorem mul_eq_zero_of_ne_zero_imp_eq_zero {a b : M₀} (h : a ≠ 0 → b = 0) : a * b = 0 := by
  /-
    M₀ : Type u_1
    inst✝ : MulZeroClass M₀
    a b : M₀
    h : Ne a 0 → Eq b 0
    ⊢ Eq (HMul.hMul a b) 0
  -/
  have : Decidable (a = 0) := Classical.propDecidable (a = 0)
  /-
    M₀ : Type u_1
    inst✝ : MulZeroClass M₀
    a b : M₀
    h : Ne a 0 → Eq b 0
    this : Decidable (Eq a 0)
    ⊢ Eq (HMul.hMul a b) 0
  -/
  exact if ha : a = 0 then by rw [ha, zero_mul] else by rw [h ha, mul_zero]
  /-
    🎉 no goals
  -/


/-- To match `one_mul_eq_id`. -/
theorem zero_mul_eq_const : ((0 : M₀) * ·) = Function.const _ 0 :=
  funext zero_mul


/-- To match `mul_one_eq_id`. -/
theorem mul_zero_eq_const : (· * (0 : M₀)) = Function.const _ 0 :=
  funext mul_zero


theorem eq_zero_of_mul_self_eq_zero (h : a * a = 0) : a = 0 :=
  (eq_zero_or_eq_zero_of_mul_eq_zero h).elim id id


@[field_simps]
theorem mul_ne_zero (ha : a ≠ 0) (hb : b ≠ 0) : a * b ≠ 0 :=
  mt eq_zero_or_eq_zero_of_mul_eq_zero <| not_or.mpr ⟨ha, hb⟩


instance mul [Zero M₀] [Mul M₀] [NoZeroDivisors M₀] {x y : M₀} [NeZero x] [NeZero y] :
    NeZero (x * y) :=
  ⟨mul_ne_zero out out⟩


/-- In a monoid with zero, if zero equals one, then zero is the only element. -/
theorem eq_zero_of_zero_eq_one (h : (0 : M₀) = 1) (a : M₀) : a = 0 := by
  /-
    M₀ : Type u_1
    inst✝ : MulZeroOneClass M₀
    h : Eq 0 1
    a : M₀
    ⊢ Eq a 0
  -/
  rw [← mul_one a, ← h, mul_zero]
  /-
    🎉 no goals
  -/


/-- In a monoid with zero, if zero equals one, then zero is the unique element.

Somewhat arbitrarily, we define the default element to be `0`.
All other elements will be provably equal to it, but not necessarily definitionally equal. -/
def uniqueOfZeroEqOne (h : (0 : M₀) = 1) : Unique M₀ where
  default := 0
  uniq := eq_zero_of_zero_eq_one h


/-- In a monoid with zero, zero equals one if and only if all elements of that semiring
are equal. -/
theorem subsingleton_iff_zero_eq_one : (0 : M₀) = 1 ↔ Subsingleton M₀ :=
  ⟨fun h => haveI := uniqueOfZeroEqOne h; inferInstance, fun h => @Subsingleton.elim _ h _ _⟩


alias ⟨subsingleton_of_zero_eq_one, _⟩ := subsingleton_iff_zero_eq_one


theorem eq_of_zero_eq_one (h : (0 : M₀) = 1) (a b : M₀) : a = b :=
  @Subsingleton.elim _ (subsingleton_of_zero_eq_one h) a b


/-- In a monoid with zero, either zero and one are nonequal, or zero is the only element. -/
theorem zero_ne_one_or_forall_eq_0 : (0 : M₀) ≠ 1 ∨ ∀ a : M₀, a = 0 :=
  not_or_of_imp eq_zero_of_zero_eq_one


theorem left_ne_zero_of_mul_eq_one (h : a * b = 1) : a ≠ 0 :=
  left_ne_zero_of_mul <| ne_zero_of_eq_one h


theorem right_ne_zero_of_mul_eq_one (h : a * b = 1) : b ≠ 0 :=
  right_ne_zero_of_mul <| ne_zero_of_eq_one h


@[simp] lemma zero_pow : ∀ {n : ℕ}, n ≠ 0 → (0 : M₀) ^ n = 0
                   /-
                     M₀ : Type u_1
                     inst✝ : MonoidWithZero M₀
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 1) 0
                     ⊢ Eq (HPow.hPow 0 (HAdd.hAdd n 1)) 0
                   -/
  | n + 1, _ => by rw [pow_succ, mul_zero]
                   /-
                     🎉 no goals
                   -/


lemma zero_pow_eq (n : ℕ) : (0 : M₀) ^ n = if n = 0 then 1 else 0 := by
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    n : Nat
    ⊢ Eq (HPow.hPow 0 n) (ite (Eq n 0) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      n : Nat
      h : Eq n 0
      ⊢ Eq (HPow.hPow 0 n) 1
    -/
  · rw [h, pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      M₀ : Type u_1
      inst✝ : MonoidWithZero M₀
      n : Nat
      h : Not (Eq n 0)
      ⊢ Eq (HPow.hPow 0 n) 0
    -/
  · rw [zero_pow h]
    /-
      🎉 no goals
    -/


lemma zero_pow_eq_one₀ [Nontrivial M₀] : (0 : M₀) ^ n = 1 ↔ n = 0 := by
  /-
    M₀ : Type u_1
    inst✝¹ : MonoidWithZero M₀
    n : Nat
    inst✝ : Nontrivial M₀
    ⊢ Iff (Eq (HPow.hPow 0 n) 1) (Eq n 0)
  -/
  rw [zero_pow_eq, one_ne_zero.ite_eq_left_iff]
  /-
    🎉 no goals
  -/


lemma pow_eq_zero_of_le : ∀ {m n} (_ : m ≤ n) (_ : a ^ m = 0), a ^ n = 0
  | _, _, Nat.le.refl, ha => ha
                                    /-
                                      M₀ : Type u_1
                                      inst✝ : MonoidWithZero M₀
                                      a : M₀
                                      n✝ m✝ : Nat
                                      hmn : n✝.le m✝
                                      ha : Eq (HPow.hPow a n✝) 0
                                      ⊢ Eq (HPow.hPow a m✝.succ) 0
                                    -/
  | _, _, Nat.le.step hmn, ha => by rw [pow_succ, pow_eq_zero_of_le hmn ha, zero_mul]
                                    /-
                                      🎉 no goals
                                    -/


                                                              /-
                                                                M₀ : Type u_1
                                                                inst✝ : MonoidWithZero M₀
                                                                a : M₀
                                                                n : Nat
                                                                hn : Ne n 0
                                                                ha : Ne (HPow.hPow a n) 0
                                                                ⊢ Ne a 0
                                                              -/
lemma ne_zero_pow (hn : n ≠ 0) (ha : a ^ n ≠ 0) : a ≠ 0 := by rintro rfl; exact ha <| zero_pow hn
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma zero_pow_eq_zero [Nontrivial M₀] : (0 : M₀) ^ n = 0 ↔ n ≠ 0 :=
      /-
        M₀ : Type u_1
        inst✝¹ : MonoidWithZero M₀
        n : Nat
        inst✝ : Nontrivial M₀
        ⊢ Eq (HPow.hPow 0 n) 0 → Ne n 0
      -/
  ⟨by rintro h rfl; simp at h, zero_pow⟩
                    /-
                      🎉 no goals
                    -/


lemma pow_mul_eq_zero_of_le {a b : M₀} {m n : ℕ} (hmn : m ≤ n)
    (h : a ^ m * b = 0) : a ^ n * b = 0 := by
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    a b : M₀
    m n : Nat
    hmn : LE.le m n
    h : Eq (HMul.hMul (HPow.hPow a m) b) 0
    ⊢ Eq (HMul.hMul (HPow.hPow a n) b) 0
  -/
  rw [show n = n - m + m by omega, pow_add, mul_assoc, h]
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    a b : M₀
    m n : Nat
    hmn : LE.le m n
    h : Eq (HMul.hMul (HPow.hPow a m) b) 0
    ⊢ Eq (HMul.hMul (HPow.hPow a (HSub.hSub n m)) 0) 0
  -/
  simp
  /-
    🎉 no goals
  -/


lemma pow_eq_zero : ∀ {n}, a ^ n = 0 → a = 0
                /-
                  M₀ : Type u_1
                  inst✝¹ : MonoidWithZero M₀
                  a : M₀
                  inst✝ : NoZeroDivisors M₀
                  ha : Eq (HPow.hPow a 0) 0
                  ⊢ Eq a 0
                -/
  | 0, ha => by simpa using congr_arg (a * ·) ha
                /-
                  🎉 no goals
                -/
                    /-
                      M₀ : Type u_1
                      inst✝¹ : MonoidWithZero M₀
                      a : M₀
                      inst✝ : NoZeroDivisors M₀
                      n : Nat
                      ha : Eq (HPow.hPow a (HAdd.hAdd n 1)) 0
                      ⊢ Eq a 0
                    -/
  | n + 1, ha => by rw [pow_succ, mul_eq_zero] at ha; exact ha.elim pow_eq_zero id
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] lemma pow_eq_zero_iff (hn : n ≠ 0) : a ^ n = 0 ↔ a = 0 :=
                   /-
                     M₀ : Type u_1
                     inst✝¹ : MonoidWithZero M₀
                     a : M₀
                     n : Nat
                     inst✝ : NoZeroDivisors M₀
                     hn : Ne n 0
                     ⊢ Eq a 0 → Eq (HPow.hPow a n) 0
                   -/
  ⟨pow_eq_zero, by rintro rfl; exact zero_pow hn⟩
                               /-
                                 🎉 no goals
                               -/


lemma pow_ne_zero_iff (hn : n ≠ 0) : a ^ n ≠ 0 ↔ a ≠ 0 := (pow_eq_zero_iff hn).not


@[field_simps]
lemma pow_ne_zero (n : ℕ) (h : a ≠ 0) : a ^ n ≠ 0 := mt pow_eq_zero h


instance NeZero.pow [NeZero a] : NeZero (a ^ n) := ⟨pow_ne_zero n NeZero.out⟩


lemma sq_eq_zero_iff : a ^ 2 = 0 ↔ a = 0 := pow_eq_zero_iff two_ne_zero


@[simp] lemma pow_eq_zero_iff' [Nontrivial M₀] : a ^ n = 0 ↔ a = 0 ∧ n ≠ 0 := by
  /-
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    a : M₀
    n : Nat
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    ⊢ Iff (Eq (HPow.hPow a n) 0) (And (Eq a 0) (Ne n 0))
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | hn := eq_or_ne n 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


instance (priority := 10) CancelMonoidWithZero.to_noZeroDivisors : NoZeroDivisors M₀ :=
  ⟨fun ab0 => or_iff_not_imp_left.mpr fun ha => mul_left_cancel₀ ha <|
    ab0.trans (mul_zero _).symm⟩


@[simp]
theorem mul_eq_mul_right_iff : a * c = b * c ↔ a = b ∨ c = 0 := by
  /-
    M₀ : Type u_1
    inst✝ : CancelMonoidWithZero M₀
    a b c : M₀
    ⊢ Iff (Eq (HMul.hMul a c) (HMul.hMul b c)) (Or (Eq a b) (Eq c 0))
  -/
  by_cases hc : c = 0 <;> [simp only [hc, mul_zero, or_true]; simp [mul_left_inj', hc]]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_eq_mul_left_iff : a * b = a * c ↔ b = c ∨ a = 0 := by
  /-
    M₀ : Type u_1
    inst✝ : CancelMonoidWithZero M₀
    a b c : M₀
    ⊢ Iff (Eq (HMul.hMul a b) (HMul.hMul a c)) (Or (Eq b c) (Eq a 0))
  -/
  by_cases ha : a = 0 <;> [simp only [ha, zero_mul, or_true]; simp [mul_right_inj', ha]]
  /-
    🎉 no goals
  -/


theorem mul_right_eq_self₀ : a * b = a ↔ b = 1 ∨ a = 0 :=
  calc
                                    /-
                                      M₀ : Type u_1
                                      inst✝ : CancelMonoidWithZero M₀
                                      a b : M₀
                                      ⊢ Iff (Eq (HMul.hMul a b) a) (Eq (HMul.hMul a b) (HMul.hMul a 1))
                                    -/
    a * b = a ↔ a * b = a * 1 := by rw [mul_one]
                                    /-
                                      🎉 no goals
                                    -/
    _ ↔ b = 1 ∨ a = 0 := mul_eq_mul_left_iff


theorem mul_left_eq_self₀ : a * b = b ↔ a = 1 ∨ b = 0 :=
  calc
                                    /-
                                      M₀ : Type u_1
                                      inst✝ : CancelMonoidWithZero M₀
                                      a b : M₀
                                      ⊢ Iff (Eq (HMul.hMul a b) b) (Eq (HMul.hMul a b) (HMul.hMul 1 b))
                                    -/
    a * b = b ↔ a * b = 1 * b := by rw [one_mul]
                                    /-
                                      🎉 no goals
                                    -/
    _ ↔ a = 1 ∨ b = 0 := mul_eq_mul_right_iff


@[simp]
theorem mul_eq_left₀ (ha : a ≠ 0) : a * b = a ↔ b = 1 := by
  /-
    M₀ : Type u_1
    inst✝ : CancelMonoidWithZero M₀
    a b : M₀
    ha : Ne a 0
    ⊢ Iff (Eq (HMul.hMul a b) a) (Eq b 1)
  -/
  rw [Iff.comm, ← mul_right_inj' ha, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_eq_right₀ (hb : b ≠ 0) : a * b = b ↔ a = 1 := by
  /-
    M₀ : Type u_1
    inst✝ : CancelMonoidWithZero M₀
    a b : M₀
    hb : Ne b 0
    ⊢ Iff (Eq (HMul.hMul a b) b) (Eq a 1)
  -/
  rw [Iff.comm, ← mul_left_inj' hb, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
                                                            /-
                                                              M₀ : Type u_1
                                                              inst✝ : CancelMonoidWithZero M₀
                                                              a b : M₀
                                                              ha : Ne a 0
                                                              ⊢ Iff (Eq a (HMul.hMul a b)) (Eq b 1)
                                                            -/
theorem left_eq_mul₀ (ha : a ≠ 0) : a = a * b ↔ b = 1 := by rw [eq_comm, mul_eq_left₀ ha]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
                                                             /-
                                                               M₀ : Type u_1
                                                               inst✝ : CancelMonoidWithZero M₀
                                                               a b : M₀
                                                               hb : Ne b 0
                                                               ⊢ Iff (Eq b (HMul.hMul a b)) (Eq a 1)
                                                             -/
theorem right_eq_mul₀ (hb : b ≠ 0) : b = a * b ↔ a = 1 := by rw [eq_comm, mul_eq_right₀ hb]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- An element of a `CancelMonoidWithZero` fixed by right multiplication by an element other
than one must be zero. -/
theorem eq_zero_of_mul_eq_self_right (h₁ : b ≠ 1) (h₂ : a * b = a) : a = 0 :=
  Classical.byContradiction fun ha => h₁ <| mul_left_cancel₀ ha <| h₂.symm ▸ (mul_one a).symm


/-- An element of a `CancelMonoidWithZero` fixed by left multiplication by an element other
than one must be zero. -/
theorem eq_zero_of_mul_eq_self_left (h₁ : b ≠ 1) (h₂ : b * a = a) : a = 0 :=
  Classical.byContradiction fun ha => h₁ <| mul_right_cancel₀ ha <| h₂.symm ▸ (one_mul a).symm


theorem GroupWithZero.mul_right_injective (h : x ≠ 0) :
    Function.Injective fun y => x * y := fun y y' w => by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    x : G₀
    h : Ne x 0
    y y' : G₀
    w : Eq ((fun y => HMul.hMul x y) y) ((fun y => HMul.hMul x y) y')
    ⊢ Eq y y'
  -/
  simpa only [← mul_assoc, inv_mul_cancel₀ h, one_mul] using congr_arg (fun y => x⁻¹ * y) w
  /-
    🎉 no goals
  -/


theorem GroupWithZero.mul_left_injective (h : x ≠ 0) :
    Function.Injective fun y => y * x := fun y y' w => by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    x : G₀
    h : Ne x 0
    y y' : G₀
    w : Eq ((fun y => HMul.hMul y x) y) ((fun y => HMul.hMul y x) y')
    ⊢ Eq y y'
  -/
  simpa only [mul_assoc, mul_inv_cancel₀ h, mul_one] using congr_arg (fun y => y * x⁻¹) w
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_mul_cancel_right₀ (h : b ≠ 0) (a : G₀) : a * b⁻¹ * b = a :=
  calc
    a * b⁻¹ * b = a * (b⁻¹ * b) := mul_assoc _ _ _
                /-
                  G₀ : Type u_2
                  inst✝ : GroupWithZero G₀
                  b : G₀
                  h : Ne b 0
                  a : G₀
                  ⊢ Eq (HMul.hMul a (HMul.hMul (Inv.inv b) b)) a
                -/
    _ = a := by simp [h]
                /-
                  🎉 no goals
                -/



@[simp]
theorem inv_mul_cancel_left₀ (h : a ≠ 0) (b : G₀) : a⁻¹ * (a * b) = b :=
  calc
    a⁻¹ * (a * b) = a⁻¹ * a * b := (mul_assoc _ _ _).symm
                /-
                  G₀ : Type u_2
                  inst✝ : GroupWithZero G₀
                  a : G₀
                  h : Ne a 0
                  b : G₀
                  ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) b) b
                -/
    _ = b := by simp [h]
                /-
                  🎉 no goals
                -/



private theorem inv_eq_of_mul (h : a * b = 1) : a⁻¹ = b := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a b : G₀
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq (Inv.inv a) b
  -/
  rw [← inv_mul_cancel_left₀ (left_ne_zero_of_mul_eq_one h) b, h, mul_one]
  /-
    🎉 no goals
  -/

-- See note [lower instance priority]

instance (priority := 100) GroupWithZero.toDivisionMonoid : DivisionMonoid G₀ :=
  { ‹GroupWithZero G₀› with
    inv := Inv.inv,
    inv_inv := fun a => by
      /-
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a✝ b x a : G₀
        ⊢ Eq (Inv.inv (Inv.inv a)) a
      -/
      by_cases h : a = 0
        /-
          case pos
          M₀ : Type u_1
          G₀ : Type u_2
          inst✝ : GroupWithZero G₀
          a✝ b x a : G₀
          h : Eq a 0
          ⊢ Eq (Inv.inv (Inv.inv a)) a
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
        /-
          case neg
          M₀ : Type u_1
          G₀ : Type u_2
          inst✝ : GroupWithZero G₀
          a✝ b x a : G₀
          h : Not (Eq a 0)
          ⊢ Eq (Inv.inv (Inv.inv a)) a
        -/
      · exact left_inv_eq_right_inv (inv_mul_cancel₀ <| inv_ne_zero h) (inv_mul_cancel₀ h)
        /-
          🎉 no goals
        -/
        ,
    mul_inv_rev := fun a b => by
      /-
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a✝ b✝ x a b : G₀
        ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (Inv.inv a))
      -/
      by_cases ha : a = 0
        /-
          case pos
          M₀ : Type u_1
          G₀ : Type u_2
          inst✝ : GroupWithZero G₀
          a✝ b✝ x a b : G₀
          ha : Eq a 0
          ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (Inv.inv a))
        -/
      · simp [ha]
        /-
          🎉 no goals
        -/
      /-
        case neg
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a✝ b✝ x a b : G₀
        ha : Not (Eq a 0)
        ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (Inv.inv a))
      -/
      by_cases hb : b = 0
        /-
          case pos
          M₀ : Type u_1
          G₀ : Type u_2
          inst✝ : GroupWithZero G₀
          a✝ b✝ x a b : G₀
          ha : Not (Eq a 0)
          hb : Eq b 0
          ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (Inv.inv a))
        -/
      · simp [hb]
        /-
          🎉 no goals
        -/
      /-
        case neg
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a✝ b✝ x a b : G₀
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        ⊢ Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (Inv.inv a))
      -/
      apply inv_eq_of_mul
      /-
        case neg.h
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a✝ b✝ x a b : G₀
        ha : Not (Eq a 0)
        hb : Not (Eq b 0)
        ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul (Inv.inv b) (Inv.inv a))) 1
      -/
      simp [mul_assoc, ha, hb],
      /-
        🎉 no goals
      -/
    inv_eq_of_mul := fun _ _ => inv_eq_of_mul }

-- see Note [lower instance priority]

instance (priority := 10) GroupWithZero.toCancelMonoidWithZero : CancelMonoidWithZero G₀ :=
  { (‹_› : GroupWithZero G₀) with
    mul_left_cancel_of_ne_zero := @fun x y z hx h => by
      /-
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a b x✝ x y z : G₀
        hx : Ne x 0
        h : Eq (HMul.hMul x y) (HMul.hMul x z)
        ⊢ Eq y z
      -/
      rw [← inv_mul_cancel_left₀ hx y, h, inv_mul_cancel_left₀ hx z],
      /-
        🎉 no goals
      -/
    mul_right_cancel_of_ne_zero := @fun x y z hy h => by
      /-
        M₀ : Type u_1
        G₀ : Type u_2
        inst✝ : GroupWithZero G₀
        a b x✝ x y z : G₀
        hy : Ne y 0
        h : Eq (HMul.hMul x y) (HMul.hMul z y)
        ⊢ Eq x z
      -/
      rw [← mul_inv_cancel_right₀ hy x, h, mul_inv_cancel_right₀ hy z] }
      /-
        🎉 no goals
      -/


@[simp]
                                            /-
                                              G₀ : Type u_2
                                              inst✝ : GroupWithZero G₀
                                              a : G₀
                                              ⊢ Eq (HDiv.hDiv 0 a) 0
                                            -/
theorem zero_div (a : G₀) : 0 / a = 0 := by rw [div_eq_mul_inv, zero_mul]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                            /-
                                              G₀ : Type u_2
                                              inst✝ : GroupWithZero G₀
                                              a : G₀
                                              ⊢ Eq (HDiv.hDiv a 0) 0
                                            -/
theorem div_zero (a : G₀) : a / 0 = 0 := by rw [div_eq_mul_inv, inv_zero, mul_zero]
                                            /-
                                              🎉 no goals
                                            -/


/-- Multiplying `a` by itself and then by its inverse results in `a`
(whether or not `a` is zero). -/
@[simp]
theorem mul_self_mul_inv (a : G₀) : a * a * a⁻¹ = a := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (HMul.hMul (HMul.hMul a a) (Inv.inv a)) a
  -/
  by_cases h : a = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Eq a 0
      ⊢ Eq (HMul.hMul (HMul.hMul a a) (Inv.inv a)) a
    -/
  · rw [h, inv_zero, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HMul.hMul a a) (Inv.inv a)) a
    -/
  · rw [mul_assoc, mul_inv_cancel₀ h, mul_one]
    /-
      🎉 no goals
    -/



/-- Multiplying `a` by its inverse and then by itself results in `a`
(whether or not `a` is zero). -/
@[simp]
theorem mul_inv_mul_cancel (a : G₀) : a * a⁻¹ * a = a := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv a)) a) a
  -/
  by_cases h : a = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Eq a 0
      ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv a)) a) a
    -/
  · rw [h, inv_zero, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HMul.hMul a (Inv.inv a)) a) a
    -/
  · rw [mul_inv_cancel₀ h, one_mul]
    /-
      🎉 no goals
    -/



/-- Multiplying `a⁻¹` by `a` twice results in `a` (whether or not `a`
is zero). -/
@[simp]
theorem inv_mul_mul_self (a : G₀) : a⁻¹ * a * a = a := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) a) a
  -/
  by_cases h : a = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Eq a 0
      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) a) a
    -/
  · rw [h, inv_zero, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) a) a
    -/
  · rw [inv_mul_cancel₀ h, one_mul]
    /-
      🎉 no goals
    -/



/-- Multiplying `a` by itself and then dividing by itself results in `a`, whether or not `a` is
zero. -/
@[simp]
                                                         /-
                                                           G₀ : Type u_2
                                                           inst✝ : GroupWithZero G₀
                                                           a : G₀
                                                           ⊢ Eq (HDiv.hDiv (HMul.hMul a a) a) a
                                                         -/
theorem mul_self_div_self (a : G₀) : a * a / a = a := by rw [div_eq_mul_inv, mul_self_mul_inv a]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Dividing `a` by itself and then multiplying by itself results in `a`, whether or not `a` is
zero. -/
@[simp]
                                                         /-
                                                           G₀ : Type u_2
                                                           inst✝ : GroupWithZero G₀
                                                           a : G₀
                                                           ⊢ Eq (HMul.hMul (HDiv.hDiv a a) a) a
                                                         -/
theorem div_self_mul_self (a : G₀) : a / a * a = a := by rw [div_eq_mul_inv, mul_inv_mul_cancel a]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem div_self_mul_self' (a : G₀) : a / (a * a) = a⁻¹ :=
  calc
                                          /-
                                            G₀ : Type u_2
                                            inst✝ : GroupWithZero G₀
                                            a : G₀
                                            ⊢ Eq (HDiv.hDiv a (HMul.hMul a a)) (HMul.hMul (HMul.hMul (Inv.inv (Inv.inv a)) …
                                          -/
    a / (a * a) = a⁻¹⁻¹ * a⁻¹ * a⁻¹ := by simp [mul_inv_rev]
                                          /-
                                            🎉 no goals
                                          -/
    _ = a⁻¹ := inv_mul_mul_self _



theorem one_div_ne_zero {a : G₀} (h : a ≠ 0) : 1 / a ≠ 0 := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    h : Ne a 0
    ⊢ Ne (HDiv.hDiv 1 a) 0
  -/
  simpa only [one_div] using inv_ne_zero h
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       G₀ : Type u_2
                                                       inst✝ : GroupWithZero G₀
                                                       a : G₀
                                                       ⊢ Iff (Eq (Inv.inv a) 0) (Eq a 0)
                                                     -/
theorem inv_eq_zero {a : G₀} : a⁻¹ = 0 ↔ a = 0 := by rw [inv_eq_iff_eq_inv, inv_zero]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem zero_eq_inv {a : G₀} : 0 = a⁻¹ ↔ 0 = a :=
  eq_comm.trans <| inv_eq_zero.trans eq_comm


/-- Dividing `a` by the result of dividing `a` by itself results in
`a` (whether or not `a` is zero). -/
@[simp]
theorem div_div_self (a : G₀) : a / (a / a) = a := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (HDiv.hDiv a (HDiv.hDiv a a)) a
  -/
  rw [div_div_eq_mul_div]
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (HDiv.hDiv (HMul.hMul a a) a) a
  -/
  exact mul_self_div_self a
  /-
    🎉 no goals
  -/


theorem ne_zero_of_one_div_ne_zero {a : G₀} (h : 1 / a ≠ 0) : a ≠ 0 := fun ha : a = 0 => by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    h : Ne (HDiv.hDiv 1 a) 0
    ha : Eq a 0
    ⊢ False
  -/
  rw [ha, div_zero] at h
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    h : Ne 0 0
    ha : Eq a 0
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


theorem eq_zero_of_one_div_eq_zero {a : G₀} (h : 1 / a = 0) : a = 0 :=
  Classical.byCases (fun ha => ha) fun ha => ((one_div_ne_zero ha) h).elim


theorem mul_left_surjective₀ {a : G₀} (h : a ≠ 0) : Surjective fun g => a * g := fun g =>
               /-
                 G₀ : Type u_2
                 inst✝ : GroupWithZero G₀
                 a : G₀
                 h : Ne a 0
                 g : G₀
                 ⊢ Eq ((fun g => HMul.hMul a g) (HMul.hMul (Inv.inv a) g)) g
               -/
  ⟨a⁻¹ * g, by simp [← mul_assoc, mul_inv_cancel₀ h]⟩
               /-
                 🎉 no goals
               -/


theorem mul_right_surjective₀ {a : G₀} (h : a ≠ 0) : Surjective fun g => g * a := fun g =>
               /-
                 G₀ : Type u_2
                 inst✝ : GroupWithZero G₀
                 a : G₀
                 h : Ne a 0
                 g : G₀
                 ⊢ Eq ((fun g => HMul.hMul g a) (HMul.hMul g (Inv.inv a))) g
               -/
  ⟨g * a⁻¹, by simp [mul_assoc, inv_mul_cancel₀ h]⟩
               /-
                 🎉 no goals
               -/


lemma zero_zpow : ∀ n : ℤ, n ≠ 0 → (0 : G₀) ^ n = 0
                     /-
                       G₀ : Type u_2
                       inst✝ : GroupWithZero G₀
                       n : Nat
                       h : Ne (↑n) 0
                       ⊢ Eq (HPow.hPow 0 ↑n) 0
                     -/
  | (n : ℕ), h => by rw [zpow_natCast, zero_pow]; simpa [Int.natCast_eq_zero] using h
                                                  /-
                                                    🎉 no goals
                                                  -/
                        /-
                          G₀ : Type u_2
                          inst✝ : GroupWithZero G₀
                          n : Nat
                          x✝ : Ne (Int.negSucc n) 0
                          ⊢ Eq (HPow.hPow 0 (Int.negSucc n)) 0
                        -/
  | .negSucc n, _ => by simp
                        /-
                          🎉 no goals
                        -/


lemma zero_zpow_eq (n : ℤ) : (0 : G₀) ^ n = if n = 0 then 1 else 0 := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    n : Int
    ⊢ Eq (HPow.hPow 0 n) (ite (Eq n 0) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      n : Int
      h : Eq n 0
      ⊢ Eq (HPow.hPow 0 n) 1
    -/
  · rw [h, zpow_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      n : Int
      h : Not (Eq n 0)
      ⊢ Eq (HPow.hPow 0 n) 0
    -/
  · rw [zero_zpow _ h]
    /-
      🎉 no goals
    -/


lemma zero_zpow_eq_one₀ {n : ℤ} : (0 : G₀) ^ n = 1 ↔ n = 0 := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    n : Int
    ⊢ Iff (Eq (HPow.hPow 0 n) 1) (Eq n 0)
  -/
  rw [zero_zpow_eq, one_ne_zero.ite_eq_left_iff]
  /-
    🎉 no goals
  -/


lemma zpow_add_one₀ (ha : a ≠ 0) : ∀ n : ℤ, a ^ (n + 1) = a ^ n * a
                  /-
                    G₀ : Type u_2
                    inst✝ : GroupWithZero G₀
                    a : G₀
                    ha : Ne a 0
                    n : Nat
                    ⊢ Eq (HPow.hPow a (HAdd.hAdd (↑n) 1)) (HMul.hMul (HPow.hPow a ↑n) a)
                  -/
  | (n : ℕ) => by simp only [← Int.ofNat_succ, zpow_natCast, pow_succ]
                  /-
                    🎉 no goals
                  -/
                     /-
                       G₀ : Type u_2
                       inst✝ : GroupWithZero G₀
                       a : G₀
                       ha : Ne a 0
                       ⊢ Eq (HPow.hPow a (HAdd.hAdd (Int.negSucc 0) 1)) (HMul.hMul (HPow.hPow a (Int. …
                     -/
  | .negSucc 0 => by erw [zpow_zero, zpow_negSucc, pow_one, inv_mul_cancel₀ ha]
                     /-
                       🎉 no goals
                     -/
  | .negSucc (n + 1) => by
    rw [Int.negSucc_eq, zpow_neg, Int.neg_add, Int.neg_add_cancel_right, zpow_neg, ← Int.ofNat_succ,
      zpow_natCast, zpow_natCast, pow_succ' _ (n + 1), mul_inv_rev, mul_assoc, inv_mul_cancel₀ ha,
      mul_one]


lemma zpow_sub_one₀ (ha : a ≠ 0) (n : ℤ) : a ^ (n - 1) = a ^ n * a⁻¹ :=
  calc
                                              /-
                                                G₀ : Type u_2
                                                inst✝ : GroupWithZero G₀
                                                a : G₀
                                                ha : Ne a 0
                                                n : Int
                                                ⊢ Eq (HPow.hPow a (HSub.hSub n 1)) (HMul.hMul (HMul.hMul (HPow.hPow a (HSub.hS …
                                              -/
    a ^ (n - 1) = a ^ (n - 1) * a * a⁻¹ := by rw [mul_assoc, mul_inv_cancel₀ ha, mul_one]
                                              /-
                                                🎉 no goals
                                              -/
                          /-
                            G₀ : Type u_2
                            inst✝ : GroupWithZero G₀
                            a : G₀
                            ha : Ne a 0
                            n : Int
                            ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a (HSub.hSub n 1)) a) (Inv.inv a)) (HMul …
                          -/
    _ = a ^ n * a⁻¹ := by rw [← zpow_add_one₀ ha, Int.sub_add_cancel]
                          /-
                            🎉 no goals
                          -/


lemma zpow_add₀ (ha : a ≠ 0) (m n : ℤ) : a ^ (m + n) = a ^ m * a ^ n := by
  induction n using Int.induction_on with
  | hz => simp
  | hp n ihn => simp only [← Int.add_assoc, zpow_add_one₀ ha, ihn, mul_assoc]
  | hn n ihn => rw [zpow_sub_one₀ ha, ← mul_assoc, ← ihn, ← zpow_sub_one₀ ha, Int.add_sub_assoc]


lemma zpow_add' {m n : ℤ} (h : a ≠ 0 ∨ m + n ≠ 0 ∨ m = 0 ∧ n = 0) :
    a ^ (m + n) = a ^ m * a ^ n := by
  /-
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    m n : Int
    h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
    ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
  -/
  by_cases hm : m = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      m n : Int
      h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
      hm : Eq m 0
      ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
    -/
  · simp [hm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    m n : Int
    h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
    hm : Not (Eq m 0)
    ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
  -/
  by_cases hn : n = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      m n : Int
      h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
      hm : Not (Eq m 0)
      hn : Eq n 0
      ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    G₀ : Type u_2
    inst✝ : GroupWithZero G₀
    a : G₀
    m n : Int
    h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
    hm : Not (Eq m 0)
    hn : Not (Eq n 0)
    ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
  -/
  by_cases ha : a = 0
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      m n : Int
      h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
      hm : Not (Eq m 0)
      hn : Not (Eq n 0)
      ha : Eq a 0
      ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
    -/
  · subst a
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      m n : Int
      hm : Not (Eq m 0)
      hn : Not (Eq n 0)
      h : Or (Ne 0 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow 0 m) (HPow.hPow 0 n))
    -/
    simp only [false_or, eq_self_iff_true, not_true, Ne, hm, hn, false_and, or_false] at h
    /-
      case pos
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      m n : Int
      hm : Not (Eq m 0)
      hn : Not (Eq n 0)
      h : Not (Eq (HAdd.hAdd m n) 0)
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow 0 m) (HPow.hPow 0 n))
    -/
    rw [zero_zpow _ h, zero_zpow _ hm, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_2
      inst✝ : GroupWithZero G₀
      a : G₀
      m n : Int
      h : Or (Ne a 0) (Or (Ne (HAdd.hAdd m n) 0) (And (Eq m 0) (Eq n 0)))
      hm : Not (Eq m 0)
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HPow.hPow a (HAdd.hAdd m n)) (HMul.hMul (HPow.hPow a m) (HPow.hPow a n))
    -/
  · exact zpow_add₀ ha m n
    /-
      🎉 no goals
    -/


                                                                        /-
                                                                          G₀ : Type u_2
                                                                          inst✝ : GroupWithZero G₀
                                                                          a : G₀
                                                                          h : Ne a 0
                                                                          i : Int
                                                                          ⊢ Eq (HPow.hPow a (HAdd.hAdd 1 i)) (HMul.hMul a (HPow.hPow a i))
                                                                        -/
lemma zpow_one_add₀ (h : a ≠ 0) (i : ℤ) : a ^ (1 + i) = a * a ^ i := by rw [zpow_add₀ h, zpow_one]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem div_mul_eq_mul_div₀ (a b c : G₀) : a / c * b = a * b / c := by
  /-
    G₀ : Type u_2
    inst✝ : CommGroupWithZero G₀
    a b c : G₀
    ⊢ Eq (HMul.hMul (HDiv.hDiv a c) b) (HDiv.hDiv (HMul.hMul a b) c)
  -/
  simp_rw [div_eq_mul_inv, mul_assoc, mul_comm c⁻¹]
  /-
    🎉 no goals
  -/


lemma div_sq_cancel (a b : G₀) : a ^ 2 * b / a = a * b := by
  /-
    G₀ : Type u_2
    inst✝ : CommGroupWithZero G₀
    a b : G₀
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow a 2) b) a) (HMul.hMul a b)
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      G₀ : Type u_2
      inst✝ : CommGroupWithZero G₀
      b : G₀
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow 0 2) b) 0) (HMul.hMul 0 b)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝ : CommGroupWithZero G₀
      a b : G₀
      ha : Ne a 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow a 2) b) a) (HMul.hMul a b)
    -/
  · rw [sq, mul_assoc, mul_div_cancel_left₀ _ ha]
    /-
      🎉 no goals
    -/


