@[to_additive]
instance commGroup : CommGroup PUnit where
  mul _ _ := unit
  one := unit
  inv _ := unit
  div _ _ := unit
  npow _ _ := unit
  zpow _ _ := unit
                  /-
                    ⊢ ∀ (a b c : PUnit.{?u.2 + 1}), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a  …
                  -/
  mul_assoc := by intros; rfl
                          /-
                            🎉 no goals
                          -/
                /-
                  ⊢ ∀ (a : PUnit.{?u.2 + 1}), Eq (HMul.hMul 1 a) a
                -/
  one_mul := by intros; rfl
                        /-
                          🎉 no goals
                        -/
                /-
                  ⊢ ∀ (a : PUnit.{?u.2 + 1}), Eq (HMul.hMul a 1) a
                -/
  mul_one := by intros; rfl
                        /-
                          🎉 no goals
                        -/
                       /-
                         ⊢ ∀ (a : PUnit.{?u.2 + 1}), Eq (HMul.hMul (Inv.inv a) a) 1
                       -/
  inv_mul_cancel := by intros; rfl
                               /-
                                 🎉 no goals
                               -/
                 /-
                   ⊢ ∀ (a b : PUnit.{?u.2 + 1}), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
  mul_comm := by intros; rfl
                         /-
                           🎉 no goals
                         -/

-- shortcut instances

@[to_additive] instance : One PUnit where one := ()

@[to_additive] instance : Mul PUnit where mul _ _ := ()

@[to_additive] instance : Div PUnit where div _ _ := ()

@[to_additive] instance : Inv PUnit where inv _ := ()

-- dsimp loops when applying this lemma to its LHS,
-- probably https://github.com/leanprover/lean4/pull/2867

@[to_additive (attr := simp, nolint simpNF)]
theorem one_eq : (1 : PUnit) = unit :=
  rfl

-- note simp can prove this when the Boolean ring structure is introduced

@[to_additive]
theorem mul_eq {x y : PUnit} : x * y = unit :=
  rfl


@[to_additive (attr := simp)]
theorem div_eq {x y : PUnit} : x / y = unit :=
  rfl


@[to_additive (attr := simp)]
theorem inv_eq {x : PUnit} : x⁻¹ = unit :=
  rfl


instance commRing : CommRing PUnit where
  __ := PUnit.commGroup
  __ := PUnit.addCommGroup
                     /-
                       ⊢ ∀ (a b c : PUnit.{?u.1286 + 1}), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd …
                     -/
  left_distrib := by intros; rfl
                             /-
                               🎉 no goals
                             -/
                      /-
                        ⊢ ∀ (a b c : PUnit.{?u.1286 + 1}), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd …
                      -/
  right_distrib := by intros; rfl
                              /-
                                🎉 no goals
                              -/
                 /-
                   ⊢ ∀ (a : PUnit.{?u.1286 + 1}), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by intros; rfl
                         /-
                           🎉 no goals
                         -/
                 /-
                   ⊢ ∀ (a : PUnit.{?u.1286 + 1}), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by intros; rfl
                         /-
                           🎉 no goals
                         -/
  natCast _ := unit


instance cancelCommMonoidWithZero : CancelCommMonoidWithZero PUnit where


