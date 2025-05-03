/-- An auxiliary lemma that can be used to prove `⇑(f ^ n) = ⇑f^[n]`. -/
theorem hom_coe_pow {F : Type*} [Monoid F] (c : F → M → M) (h1 : c 1 = id)
    (hmul : ∀ f g, c (f * g) = c f ∘ c g) (f : F) : ∀ n, c (f ^ n) = (c f)^[n]
  | 0 => by
    /-
      M : Type u_1
      F : Type u_4
      inst✝ : Monoid F
      c : F → M → M
      h1 : Eq (c 1) id
      hmul : ∀ (f g : F), Eq (c (HMul.hMul f g)) (Function.comp (c f) (c g))
      f : F
      ⊢ Eq (c (HPow.hPow f 0)) (Nat.iterate (c f) 0)
    -/
    rw [pow_zero, h1]
    /-
      M : Type u_1
      F : Type u_4
      inst✝ : Monoid F
      c : F → M → M
      h1 : Eq (c 1) id
      hmul : ∀ (f g : F), Eq (c (HMul.hMul f g)) (Function.comp (c f) (c g))
      f : F
      ⊢ Eq id (Nat.iterate (c f) 0)
    -/
    rfl
    /-
      🎉 no goals
    -/
                /-
                  M : Type u_1
                  F : Type u_4
                  inst✝ : Monoid F
                  c : F → M → M
                  h1 : Eq (c 1) id
                  hmul : ∀ (f g : F), Eq (c (HMul.hMul f g)) (Function.comp (c f) (c g))
                  f : F
                  n : Nat
                  ⊢ Eq (c (HPow.hPow f (HAdd.hAdd n 1))) (Nat.iterate (c f) (HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [pow_succ, iterate_succ, hmul, hom_coe_pow c h1 hmul f n]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
theorem iterate_map_mul {M F : Type*} [Mul M] [FunLike F M M] [MulHomClass F M M]
    (f : F) (n : ℕ) (x y : M) :
    f^[n] (x * y) = f^[n] x * f^[n] y :=
  Function.Semiconj₂.iterate (map_mul f) n x y


@[to_additive (attr := simp)]
theorem iterate_map_one {M F : Type*} [One M] [FunLike F M M] [OneHomClass F M M]
    (f : F) (n : ℕ) :
    f^[n] 1 = 1 :=
  iterate_fixed (map_one f) n


@[to_additive (attr := simp)]
theorem iterate_map_inv {M F : Type*} [Group M] [FunLike F M M] [MonoidHomClass F M M]
    (f : F) (n : ℕ) (x : M) :
    f^[n] x⁻¹ = (f^[n] x)⁻¹ :=
  Commute.iterate_left (map_inv f) n x


@[to_additive (attr := simp)]
theorem iterate_map_div {M F : Type*} [Group M] [FunLike F M M] [MonoidHomClass F M M]
    (f : F) (n : ℕ) (x y : M) :
    f^[n] (x / y) = f^[n] x / f^[n] y :=
  Semiconj₂.iterate (map_div f) n x y


@[to_additive (attr := simp)]
theorem iterate_map_pow {M F : Type*} [Monoid M] [FunLike F M M] [MonoidHomClass F M M]
    (f : F) (n : ℕ) (x : M) (k : ℕ) :
    f^[n] (x ^ k) = f^[n] x ^ k :=
  Commute.iterate_left (map_pow f · k) n x


@[to_additive (attr := simp)]
theorem iterate_map_zpow {M F : Type*} [Group M] [FunLike F M M] [MonoidHomClass F M M]
    (f : F) (n : ℕ) (x : M) (k : ℤ) :
    f^[n] (x ^ k) = f^[n] x ^ k :=
  Commute.iterate_left (map_zpow f · k) n x

--what should be the namespace for this section?

@[to_additive (attr := simp)]
theorem smul_iterate [MulAction G H] : (a • · : H → H)^[n] = (a ^ n • ·) :=
  funext fun b =>
                    /-
                      G : Type u_2
                      H : Type u_3
                      inst✝¹ : Monoid G
                      a : G
                      n : Nat
                      inst✝ : MulAction G H
                      b : H
                      ⊢ Eq (Nat.iterate (fun x => HSMul.hSMul a x) Nat.zero b) (HSMul.hSMul (HPow.hP …
                    -/
    Nat.recOn n (by rw [iterate_zero, id, pow_zero, one_smul])
                    /-
                      🎉 no goals
                    -/
                   /-
                     G : Type u_2
                     H : Type u_3
                     inst✝¹ : Monoid G
                     a : G
                     n✝ : Nat
                     inst✝ : MulAction G H
                     b : H
                     n : Nat
                     ih : Eq (Nat.iterate (fun x => HSMul.hSMul a x) n b) (HSMul.hSMul (HPow.hPow a …
                     ⊢ Eq (Nat.iterate (fun x => HSMul.hSMul a x) n.succ b) (HSMul.hSMul (HPow.hPow …
                   -/
    fun n ih => by rw [iterate_succ', comp_apply, ih, pow_succ', mul_smul]
                   /-
                     🎉 no goals
                   -/


@[to_additive]
lemma smul_iterate_apply [MulAction G H] {b : H} : (a • ·)^[n] b = a ^ n • b := by
  /-
    G : Type u_2
    H : Type u_3
    inst✝¹ : Monoid G
    a : G
    n : Nat
    inst✝ : MulAction G H
    b : H
    ⊢ Eq (Nat.iterate (fun x => HSMul.hSMul a x) n b) (HSMul.hSMul (HPow.hPow a n) …
  -/
  rw [smul_iterate]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_left_iterate : (a * ·)^[n] = (a ^ n * ·) :=
  smul_iterate a n


@[to_additive (attr := simp)]
theorem mul_right_iterate : (· * a)^[n] = (· * a ^ n) :=
  smul_iterate (MulOpposite.op a) n


@[to_additive]
                                                                  /-
                                                                    G : Type u_2
                                                                    inst✝ : Monoid G
                                                                    a : G
                                                                    n : Nat
                                                                    ⊢ Eq (Nat.iterate (fun x => HMul.hMul x a) n 1) (HPow.hPow a n)
                                                                  -/
theorem mul_right_iterate_apply_one : (· * a)^[n] 1 = a ^ n := by simp [mul_right_iterate]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive (attr := simp)]
theorem pow_iterate (n : ℕ) (j : ℕ) : (fun x : G => x ^ n)^[j] = fun x : G => x ^ n ^ j :=
  letI : MulAction ℕ G :=
    { smul := fun n g => g ^ n
      one_smul := pow_one
      mul_smul := fun m n g => pow_mul' g m n }
  smul_iterate n j


@[to_additive (attr := simp)]
theorem zpow_iterate (n : ℤ) (j : ℕ) : (fun x : G => x ^ n)^[j] = fun x => x ^ n ^ j :=
  letI : MulAction ℤ G :=
    { smul := fun n g => g ^ n
      one_smul := zpow_one
      mul_smul := fun m n g => zpow_mul' g m n }
  smul_iterate n j


@[to_additive]
theorem SemiconjBy.function_semiconj_mul_left (h : SemiconjBy a b c) :
    Function.Semiconj (a * ·) (b * ·) (c * ·) := fun j => by
  /-
    G : Type u_2
    inst✝ : Semigroup G
    a b c : G
    h : SemiconjBy a b c
    j : G
    ⊢ Eq ((fun x => HMul.hMul a x) ((fun x => HMul.hMul b x) j)) ((fun x => HMul.h …
  -/
  beta_reduce; rw [← mul_assoc, h.eq, mul_assoc]
               /-
                 🎉 no goals
               -/


@[to_additive]
theorem Commute.function_commute_mul_left (h : Commute a b) :
    Function.Commute (a * ·) (b * ·) :=
  SemiconjBy.function_semiconj_mul_left h


@[to_additive]
theorem SemiconjBy.function_semiconj_mul_right_swap (h : SemiconjBy a b c) :
                                                             /-
                                                               G : Type u_2
                                                               inst✝ : Semigroup G
                                                               a b c : G
                                                               h : SemiconjBy a b c
                                                               j : G
                                                               ⊢ Eq ((fun x => HMul.hMul x a) ((fun x => HMul.hMul x c) j)) ((fun x => HMul.h …
                                                             -/
    Function.Semiconj (· * a) (· * c) (· * b) := fun j => by simp_rw [mul_assoc, ← h.eq]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
theorem Commute.function_commute_mul_right (h : Commute a b) :
    Function.Commute (· * a) (· * b) :=
  SemiconjBy.function_semiconj_mul_right_swap h


