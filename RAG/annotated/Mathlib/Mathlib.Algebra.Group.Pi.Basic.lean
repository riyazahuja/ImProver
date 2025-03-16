@[to_additive]
instance instOne [∀ i, One <| f i] : One (∀ i : I, f i) :=
  ⟨fun _ => 1⟩


@[to_additive (attr := simp)]
theorem one_apply [∀ i, One <| f i] : (1 : ∀ i, f i) i = 1 :=
  rfl


@[to_additive]
theorem one_def [∀ i, One <| f i] : (1 : ∀ i, f i) = fun _ => 1 :=
  rfl


@[to_additive (attr := simp)] lemma _root_.Function.const_one [One β] : const α (1 : β) = 1 := rfl


@[to_additive (attr := simp)]
theorem one_comp [One γ] (x : α → β) : (1 : β → γ) ∘ x = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem comp_one [One β] (x : β → γ) : x ∘ (1 : α → β) = const α (x 1) :=
  rfl


@[to_additive]
instance instMul [∀ i, Mul <| f i] : Mul (∀ i : I, f i) :=
  ⟨fun f g i => f i * g i⟩


@[to_additive (attr := simp)]
theorem mul_apply [∀ i, Mul <| f i] : (x * y) i = x i * y i :=
  rfl


@[to_additive]
theorem mul_def [∀ i, Mul <| f i] : x * y = fun i => x i * y i :=
  rfl


@[to_additive (attr := simp)]
lemma _root_.Function.const_mul [Mul β] (a b : β) : const α a * const α b = const α (a * b) := rfl


@[to_additive]
theorem mul_comp [Mul γ] (x y : β → γ) (z : α → β) : (x * y) ∘ z = x ∘ z * y ∘ z :=
  rfl


@[to_additive]
instance instSMul [∀ i, SMul α <| f i] : SMul α (∀ i : I, f i) :=
  ⟨fun s x => fun i => s • x i⟩


@[to_additive existing instSMul]
instance instPow [∀ i, Pow (f i) β] : Pow (∀ i, f i) β :=
  ⟨fun x b i => x i ^ b⟩


@[to_additive (attr := simp, to_additive) (reorder := 5 6) smul_apply]
theorem pow_apply [∀ i, Pow (f i) β] (x : ∀ i, f i) (b : β) (i : I) : (x ^ b) i = x i ^ b :=
  rfl


@[to_additive (attr := to_additive) (reorder := 5 6) smul_def]
theorem pow_def [∀ i, Pow (f i) β] (x : ∀ i, f i) (b : β) : x ^ b = fun i => x i ^ b :=
  rfl


@[to_additive (attr := simp, to_additive) (reorder := 2 3, 5 6) smul_const]
lemma _root_.Function.const_pow [Pow α β] (a : α) (b : β) : const I a ^ b = const I (a ^ b) := rfl


@[to_additive (attr := to_additive) (reorder := 6 7) smul_comp]
theorem pow_comp [Pow γ α] (x : β → γ) (a : α) (y : I → β) : (x ^ a) ∘ y = x ∘ y ^ a :=
  rfl

-- Use `Pi.ofNat_apply` instead


@[to_additive]
instance instInv [∀ i, Inv <| f i] : Inv (∀ i : I, f i) :=
  ⟨fun f i => (f i)⁻¹⟩


@[to_additive (attr := simp)]
theorem inv_apply [∀ i, Inv <| f i] : x⁻¹ i = (x i)⁻¹ :=
  rfl


@[to_additive]
theorem inv_def [∀ i, Inv <| f i] : x⁻¹ = fun i => (x i)⁻¹ :=
  rfl


@[to_additive]
lemma _root_.Function.const_inv [Inv β] (a : β) : (const α a)⁻¹ = const α a⁻¹ := rfl


@[to_additive]
theorem inv_comp [Inv γ] (x : β → γ) (y : α → β) : x⁻¹ ∘ y = (x ∘ y)⁻¹ :=
  rfl


@[to_additive]
instance instDiv [∀ i, Div <| f i] : Div (∀ i : I, f i) :=
  ⟨fun f g i => f i / g i⟩


@[to_additive (attr := simp)]
theorem div_apply [∀ i, Div <| f i] : (x / y) i = x i / y i :=
  rfl


@[to_additive]
theorem div_def [∀ i, Div <| f i] : x / y = fun i => x i / y i :=
  rfl


@[to_additive]
theorem div_comp [Div γ] (x y : β → γ) (z : α → β) : (x / y) ∘ z = x ∘ z / y ∘ z :=
  rfl


@[to_additive (attr := simp)]
lemma _root_.Function.const_div [Div β] (a b : β) : const α a / const α b = const α (a / b) := rfl


@[to_additive]
instance semigroup [∀ i, Semigroup (f i)] : Semigroup (∀ i, f i) where
                  /-
                    I : Type u
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    f : I → Type v₁
                    g : I → Type v₂
                    h : I → Type v₃
                    x y : (i : I) → f i
                    i : I
                    inst✝ : (i : I) → Semigroup (f i)
                    ⊢ ∀ (a b c : (i : I) → f i), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HM …
                  -/
  mul_assoc := by intros; ext; exact mul_assoc _ _ _
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
instance commSemigroup [∀ i, CommSemigroup (f i)] : CommSemigroup (∀ i, f i) where
                 /-
                   I : Type u
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   f : I → Type v₁
                   g : I → Type v₂
                   h : I → Type v₃
                   x y : (i : I) → f i
                   i : I
                   inst✝ : (i : I) → CommSemigroup (f i)
                   ⊢ ∀ (a b : (i : I) → f i), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
  mul_comm := by intros; ext; exact mul_comm _ _
                              /-
                                🎉 no goals
                              -/


@[to_additive]
instance mulOneClass [∀ i, MulOneClass (f i)] : MulOneClass (∀ i, f i) where
                /-
                  I : Type u
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  f : I → Type v₁
                  g : I → Type v₂
                  h : I → Type v₃
                  x y : (i : I) → f i
                  i : I
                  inst✝ : (i : I) → MulOneClass (f i)
                  ⊢ ∀ (a : (i : I) → f i), Eq (HMul.hMul 1 a) a
                -/
  one_mul := by intros; ext; exact one_mul _
                             /-
                               🎉 no goals
                             -/
                /-
                  I : Type u
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  f : I → Type v₁
                  g : I → Type v₂
                  h : I → Type v₃
                  x y : (i : I) → f i
                  i : I
                  inst✝ : (i : I) → MulOneClass (f i)
                  ⊢ ∀ (a : (i : I) → f i), Eq (HMul.hMul a 1) a
                -/
  mul_one := by intros; ext; exact mul_one _
                             /-
                               🎉 no goals
                             -/


@[to_additive]
instance invOneClass [∀ i, InvOneClass (f i)] : InvOneClass (∀ i, f i) where
                /-
                  I : Type u
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  f : I → Type v₁
                  g : I → Type v₂
                  h : I → Type v₃
                  x y : (i : I) → f i
                  i : I
                  inst✝ : (i : I) → InvOneClass (f i)
                  ⊢ Eq (Inv.inv 1) 1
                -/
  inv_one := by ext; exact inv_one
                     /-
                       🎉 no goals
                     -/


@[to_additive]
instance monoid [∀ i, Monoid (f i)] : Monoid (∀ i, f i) where
  __ := semigroup
  __ := mulOneClass
  npow := fun n x i => x i ^ n
                  /-
                    I : Type u
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    f : I → Type v₁
                    g : I → Type v₂
                    h : I → Type v₃
                    x y : (i : I) → f i
                    i : I
                    inst✝ : (i : I) → Monoid (f i)
                    ⊢ ∀ (x : (i : I) → f i), Eq ((fun n x i => HPow.hPow (x i) n) 0 x) 1
                  -/
  npow_zero := by intros; ext; exact Monoid.npow_zero _
                               /-
                                 🎉 no goals
                               -/
                  /-
                    I : Type u
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    f : I → Type v₁
                    g : I → Type v₂
                    h : I → Type v₃
                    x y : (i : I) → f i
                    i : I
                    inst✝ : (i : I) → Monoid (f i)
                    ⊢ ∀ (n : Nat) (x : (i : I) → f i), Eq ((fun n x i => HPow.hPow (x i) n) (HAdd. …
                  -/
  npow_succ := by intros; ext; exact Monoid.npow_succ _ _
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
instance commMonoid [∀ i, CommMonoid (f i)] : CommMonoid (∀ i, f i) :=
  { monoid, commSemigroup with }


@[to_additive Pi.subNegMonoid]
instance divInvMonoid [∀ i, DivInvMonoid (f i)] : DivInvMonoid (∀ i, f i) where
  zpow := fun z x i => x i ^ z
                       /-
                         I : Type u
                         α : Type u_1
                         β : Type u_2
                         γ : Type u_3
                         f : I → Type v₁
                         g : I → Type v₂
                         h : I → Type v₃
                         x y : (i : I) → f i
                         i : I
                         inst✝ : (i : I) → DivInvMonoid (f i)
                         ⊢ ∀ (a b : (i : I) → f i), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
                       -/
  div_eq_mul_inv := by intros; ext; exact div_eq_mul_inv _ _
                                    /-
                                      🎉 no goals
                                    -/
                   /-
                     I : Type u
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     f : I → Type v₁
                     g : I → Type v₂
                     h : I → Type v₃
                     x y : (i : I) → f i
                     i : I
                     inst✝ : (i : I) → DivInvMonoid (f i)
                     ⊢ ∀ (a : (i : I) → f i), Eq ((fun z x i => HPow.hPow (x i) z) 0 a) 1
                   -/
  zpow_zero' := by intros; ext; exact DivInvMonoid.zpow_zero' _
                                /-
                                  🎉 no goals
                                -/
                   /-
                     I : Type u
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     f : I → Type v₁
                     g : I → Type v₂
                     h : I → Type v₃
                     x y : (i : I) → f i
                     i : I
                     inst✝ : (i : I) → DivInvMonoid (f i)
                     ⊢ ∀ (n : Nat) (a : (i : I) → f i), Eq ((fun z x i => HPow.hPow (x i) z) (↑n.su …
                   -/
  zpow_succ' := by intros; ext; exact DivInvMonoid.zpow_succ' _ _
                                /-
                                  🎉 no goals
                                -/
                  /-
                    I : Type u
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    f : I → Type v₁
                    g : I → Type v₂
                    h : I → Type v₃
                    x y : (i : I) → f i
                    i : I
                    inst✝ : (i : I) → DivInvMonoid (f i)
                    ⊢ ∀ (n : Nat) (a : (i : I) → f i), Eq ((fun z x i => HPow.hPow (x i) z) (Int.n …
                  -/
  zpow_neg' := by intros; ext; exact DivInvMonoid.zpow_neg' _ _
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
instance divInvOneMonoid [∀ i, DivInvOneMonoid (f i)] : DivInvOneMonoid (∀ i, f i) where
                /-
                  I : Type u
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  f : I → Type v₁
                  g : I → Type v₂
                  h : I → Type v₃
                  x y : (i : I) → f i
                  i : I
                  inst✝ : (i : I) → DivInvOneMonoid (f i)
                  ⊢ Eq (Inv.inv 1) 1
                -/
  inv_one := by ext; exact inv_one
                     /-
                       🎉 no goals
                     -/


@[to_additive]
instance involutiveInv [∀ i, InvolutiveInv (f i)] : InvolutiveInv (∀ i, f i) where
                /-
                  I : Type u
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  f : I → Type v₁
                  g : I → Type v₂
                  h : I → Type v₃
                  x y : (i : I) → f i
                  i : I
                  inst✝ : (i : I) → InvolutiveInv (f i)
                  ⊢ ∀ (x : (i : I) → f i), Eq (Inv.inv (Inv.inv x)) x
                -/
  inv_inv := by intros; ext; exact inv_inv _
                             /-
                               🎉 no goals
                             -/


@[to_additive]
instance divisionMonoid [∀ i, DivisionMonoid (f i)] : DivisionMonoid (∀ i, f i) where
  __ := divInvMonoid
  __ := involutiveInv
                    /-
                      I : Type u
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      f : I → Type v₁
                      g : I → Type v₂
                      h : I → Type v₃
                      x y : (i : I) → f i
                      i : I
                      inst✝ : (i : I) → DivisionMonoid (f i)
                      ⊢ ∀ (a b : (i : I) → f i), Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) …
                    -/
  mul_inv_rev := by intros; ext; exact mul_inv_rev _ _
                                 /-
                                   🎉 no goals
                                 -/
                      /-
                        I : Type u
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        f : I → Type v₁
                        g : I → Type v₂
                        h : I → Type v₃
                        x y : (i : I) → f i
                        i : I
                        inst✝ : (i : I) → DivisionMonoid (f i)
                        ⊢ ∀ (a b : (i : I) → f i), Eq (HMul.hMul a b) 1 → Eq (Inv.inv a) b
                      -/
  inv_eq_of_mul := by intros _ _ h; ext; exact DivisionMonoid.inv_eq_of_mul _ _ (congrFun h _)
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive instSubtractionCommMonoid]
instance divisionCommMonoid [∀ i, DivisionCommMonoid (f i)] : DivisionCommMonoid (∀ i, f i) :=
  { divisionMonoid, commSemigroup with }


@[to_additive]
instance group [∀ i, Group (f i)] : Group (∀ i, f i) where
                       /-
                         I : Type u
                         α : Type u_1
                         β : Type u_2
                         γ : Type u_3
                         f : I → Type v₁
                         g : I → Type v₂
                         h : I → Type v₃
                         x y : (i : I) → f i
                         i : I
                         inst✝ : (i : I) → Group (f i)
                         ⊢ ∀ (a : (i : I) → f i), Eq (HMul.hMul (Inv.inv a) a) 1
                       -/
  inv_mul_cancel := by intros; ext; exact inv_mul_cancel _
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive]
instance commGroup [∀ i, CommGroup (f i)] : CommGroup (∀ i, f i) := { group, commMonoid with }


@[to_additive] instance instIsLeftCancelMul [∀ i, Mul (f i)] [∀ i, IsLeftCancelMul (f i)] :
    IsLeftCancelMul (∀ i, f i) where
  mul_left_cancel  _ _ _ h := funext fun _ ↦ mul_left_cancel (congr_fun h _)


@[to_additive] instance instIsRightCancelMul [∀ i, Mul (f i)] [∀ i, IsRightCancelMul (f i)] :
    IsRightCancelMul (∀ i, f i) where
  mul_right_cancel  _ _ _ h := funext fun _ ↦ mul_right_cancel (congr_fun h _)


@[to_additive] instance instIsCancelMul [∀ i, Mul (f i)] [∀ i, IsCancelMul (f i)] :
    IsCancelMul (∀ i, f i) where


@[to_additive]
instance leftCancelSemigroup [∀ i, LeftCancelSemigroup (f i)] : LeftCancelSemigroup (∀ i, f i) :=
  { semigroup with mul_left_cancel := fun _ _ _ => mul_left_cancel }


@[to_additive]
instance rightCancelSemigroup [∀ i, RightCancelSemigroup (f i)] : RightCancelSemigroup (∀ i, f i) :=
  { semigroup with mul_right_cancel := fun _ _ _ => mul_right_cancel }


@[to_additive]
instance leftCancelMonoid [∀ i, LeftCancelMonoid (f i)] : LeftCancelMonoid (∀ i, f i) :=
  { leftCancelSemigroup, monoid with }


@[to_additive]
instance rightCancelMonoid [∀ i, RightCancelMonoid (f i)] : RightCancelMonoid (∀ i, f i) :=
  { rightCancelSemigroup, monoid with }


@[to_additive]
instance cancelMonoid [∀ i, CancelMonoid (f i)] : CancelMonoid (∀ i, f i) :=
  { leftCancelMonoid, rightCancelMonoid with }


@[to_additive]
instance cancelCommMonoid [∀ i, CancelCommMonoid (f i)] : CancelCommMonoid (∀ i, f i) :=
  { leftCancelMonoid, commMonoid with }


/-- The function supported at `i`, with value `x` there, and `1` elsewhere. -/
@[to_additive "The function supported at `i`, with value `x` there, and `0` elsewhere."]
def mulSingle (i : I) (x : f i) : ∀ (j : I), f j :=
  Function.update 1 i x


@[to_additive (attr := simp)]
theorem mulSingle_eq_same (i : I) (x : f i) : mulSingle i x i = x :=
  Function.update_self i x _


@[to_additive (attr := simp)]
theorem mulSingle_eq_of_ne {i i' : I} (h : i' ≠ i) (x : f i) : mulSingle i x i' = 1 :=
  Function.update_of_ne h x _


/-- Abbreviation for `mulSingle_eq_of_ne h.symm`, for ease of use by `simp`. -/
@[to_additive (attr := simp)
  "Abbreviation for `single_eq_of_ne h.symm`, for ease of use by `simp`."]
theorem mulSingle_eq_of_ne' {i i' : I} (h : i ≠ i') (x : f i) : mulSingle i x i' = 1 :=
  mulSingle_eq_of_ne h.symm x


@[to_additive (attr := simp)]
theorem mulSingle_one (i : I) : mulSingle i (1 : f i) = 1 :=
  Function.update_eq_self _ _

-- Porting note:
-- 1) Why do I have to specify the type of `mulSingle i x` explicitly?
-- 2) Why do I have to specify the type of `(1 : I → β)`?
-- 3) Removed `{β : Sort*}` as `[One β]` converts it to a type anyways.

/-- On non-dependent functions, `Pi.mulSingle` can be expressed as an `ite` -/
@[to_additive "On non-dependent functions, `Pi.single` can be expressed as an `ite`"]
theorem mulSingle_apply [One β] (i : I) (x : β) (i' : I) :
    (mulSingle i x : I → β) i' = if i' = i then x else 1 :=
  Function.update_apply (1 : I → β) i x i'

-- Porting note: Same as above.

/-- On non-dependent functions, `Pi.mulSingle` is symmetric in the two indices. -/
@[to_additive "On non-dependent functions, `Pi.single` is symmetric in the two indices."]
theorem mulSingle_comm [One β] (i : I) (x : β) (i' : I) :
    (mulSingle i x : I → β) i' = (mulSingle i' x : I → β) i := by
  /-
    I : Type u
    β : Type u_2
    inst✝¹ : DecidableEq I
    inst✝ : One β
    i : I
    x : β
    i' : I
    ⊢ Eq (Pi.mulSingle i x i') (Pi.mulSingle i' x i)
  -/
  simp [mulSingle_apply, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem apply_mulSingle (f' : ∀ i, f i → g i) (hf' : ∀ i, f' i 1 = 1) (i : I) (x : f i) (j : I) :
    f' j (mulSingle i x j) = mulSingle i (f' i x) j := by
  /-
    I : Type u
    f : I → Type v₁
    g : I → Type v₂
    inst✝² : DecidableEq I
    inst✝¹ : (i : I) → One (f i)
    inst✝ : (i : I) → One (g i)
    f' : (i : I) → f i → g i
    hf' : ∀ (i : I), Eq (f' i 1) 1
    i : I
    x : f i
    j : I
    ⊢ Eq (f' j (Pi.mulSingle i x j)) (Pi.mulSingle i (f' i x) j)
  -/
  simpa only [Pi.one_apply, hf', mulSingle] using Function.apply_update f' 1 i x j
  /-
    🎉 no goals
  -/


@[to_additive apply_single₂]
theorem apply_mulSingle₂ (f' : ∀ i, f i → g i → h i) (hf' : ∀ i, f' i 1 1 = 1) (i : I)
    (x : f i) (y : g i) (j : I) :
    f' j (mulSingle i x j) (mulSingle i y j) = mulSingle i (f' i x y) j := by
  /-
    I : Type u
    f : I → Type v₁
    g : I → Type v₂
    h : I → Type v₃
    inst✝³ : DecidableEq I
    inst✝² : (i : I) → One (f i)
    inst✝¹ : (i : I) → One (g i)
    inst✝ : (i : I) → One (h i)
    f' : (i : I) → f i → g i → h i
    hf' : ∀ (i : I), Eq (f' i 1 1) 1
    i : I
    x : f i
    y : g i
    j : I
    ⊢ Eq (f' j (Pi.mulSingle i x j) (Pi.mulSingle i y j)) (Pi.mulSingle i (f' i x  …
  -/
  by_cases h : j = i
    /-
      case pos
      I : Type u
      f : I → Type v₁
      g : I → Type v₂
      h✝ : I → Type v₃
      inst✝³ : DecidableEq I
      inst✝² : (i : I) → One (f i)
      inst✝¹ : (i : I) → One (g i)
      inst✝ : (i : I) → One (h✝ i)
      f' : (i : I) → f i → g i → h✝ i
      hf' : ∀ (i : I), Eq (f' i 1 1) 1
      i : I
      x : f i
      y : g i
      j : I
      h : Eq j i
      ⊢ Eq (f' j (Pi.mulSingle i x j) (Pi.mulSingle i y j)) (Pi.mulSingle i (f' i x  …
    -/
  · subst h
    /-
      case pos
      I : Type u
      f : I → Type v₁
      g : I → Type v₂
      h : I → Type v₃
      inst✝³ : DecidableEq I
      inst✝² : (i : I) → One (f i)
      inst✝¹ : (i : I) → One (g i)
      inst✝ : (i : I) → One (h i)
      f' : (i : I) → f i → g i → h i
      hf' : ∀ (i : I), Eq (f' i 1 1) 1
      j : I
      x : f j
      y : g j
      ⊢ Eq (f' j (Pi.mulSingle j x j) (Pi.mulSingle j y j)) (Pi.mulSingle j (f' j x  …
    -/
    simp only [mulSingle_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      f : I → Type v₁
      g : I → Type v₂
      h✝ : I → Type v₃
      inst✝³ : DecidableEq I
      inst✝² : (i : I) → One (f i)
      inst✝¹ : (i : I) → One (g i)
      inst✝ : (i : I) → One (h✝ i)
      f' : (i : I) → f i → g i → h✝ i
      hf' : ∀ (i : I), Eq (f' i 1 1) 1
      i : I
      x : f i
      y : g i
      j : I
      h : Not (Eq j i)
      ⊢ Eq (f' j (Pi.mulSingle i x j) (Pi.mulSingle i y j)) (Pi.mulSingle i (f' i x  …
    -/
  · simp only [mulSingle_eq_of_ne h, hf']
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mulSingle_op {g : I → Type*} [∀ i, One (g i)] (op : ∀ i, f i → g i)
    (h : ∀ i, op i 1 = 1) (i : I) (x : f i) :
    mulSingle i (op i x) = fun j => op j (mulSingle i x j) :=
  Eq.symm <| funext <| apply_mulSingle op h i x


@[to_additive]
theorem mulSingle_op₂ {g₁ g₂ : I → Type*} [∀ i, One (g₁ i)] [∀ i, One (g₂ i)]
    (op : ∀ i, g₁ i → g₂ i → f i) (h : ∀ i, op i 1 1 = 1) (i : I) (x₁ : g₁ i) (x₂ : g₂ i) :
    mulSingle i (op i x₁ x₂) = fun j => op j (mulSingle i x₁ j) (mulSingle i x₂ j) :=
  Eq.symm <| funext <| apply_mulSingle₂ op h i x₁ x₂


@[to_additive]
theorem mulSingle_injective (i : I) : Function.Injective (mulSingle i : f i → ∀ i, f i) :=
  Function.update_injective _ i


@[to_additive (attr := simp)]
theorem mulSingle_inj (i : I) {x y : f i} : mulSingle i x = mulSingle i y ↔ x = y :=
  (Pi.mulSingle_injective _ _).eq_iff


/-- The mapping into a product type built from maps into each component. -/
@[simp]
protected def prod (f' : ∀ i, f i) (g' : ∀ i, g i) (i : I) : f i × g i :=
  (f' i, g' i)

-- Porting note: simp now unfolds the lhs, so we are not marking these as simp.
-- @[simp]

theorem prod_fst_snd : Pi.prod (Prod.fst : α × β → α) (Prod.snd : α × β → β) = id :=
  rfl

-- Porting note: simp now unfolds the lhs, so we are not marking these as simp.
-- @[simp]

theorem prod_snd_fst : Pi.prod (Prod.snd : α × β → β) (Prod.fst : α × β → α) = Prod.swap :=
  rfl


@[to_additive]
theorem extend_one [One γ] (f : α → β) : Function.extend f (1 : α → γ) (1 : β → γ) = 1 :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       inst✝ : One γ
                       f : α → β
                       x✝ : β
                       ⊢ Eq (Function.extend f 1 1 x✝) (1 x✝)
                     -/
  funext fun _ => by apply ite_self
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem extend_mul [Mul γ] (f : α → β) (g₁ g₂ : α → γ) (e₁ e₂ : β → γ) :
    Function.extend f (g₁ * g₂) (e₁ * e₂) = Function.extend f g₁ e₁ * Function.extend f g₂ e₂ := by
  classical
  funext x
  simp only [not_exists, extend_def, Pi.mul_apply, apply_dite₂, dite_eq_ite, ite_self]
-- Porting note: The Lean3 statement was
-- `funext <| λ _, by convert (apply_dite2 (*) _ _ _ _ _).symm`
-- which converts to
-- `funext fun _ => by convert (apply_dite₂ (· * ·) _ _ _ _ _).symm`
-- However this does not work, and we're not sure why.


@[to_additive]
theorem extend_inv [Inv γ] (f : α → β) (g : α → γ) (e : β → γ) :
    Function.extend f g⁻¹ e⁻¹ = (Function.extend f g e)⁻¹ := by
  classical
  funext x
  simp only [not_exists, extend_def, Pi.inv_apply, apply_dite Inv.inv]
-- Porting note: The Lean3 statement was
-- `funext <| λ _, by convert (apply_dite has_inv.inv _ _ _).symm`
-- which converts to
-- `funext fun _ => by convert (apply_dite Inv.inv _ _ _).symm`
-- However this does not work, and we're not sure why.


@[to_additive]
theorem extend_div [Div γ] (f : α → β) (g₁ g₂ : α → γ) (e₁ e₂ : β → γ) :
    Function.extend f (g₁ / g₂) (e₁ / e₂) = Function.extend f g₁ e₁ / Function.extend f g₂ e₂ := by
  classical
  funext x
  simp [Function.extend_def, apply_dite₂]
-- Porting note: The Lean3 statement was
-- `funext <| λ _, by convert (apply_dite2 (/) _ _ _ _ _).symm`
-- which converts to
-- `funext fun _ => by convert (apply_dite₂ (· / ·) _ _ _ _ _).symm`
-- However this does not work, and we're not sure why.


lemma comp_eq_const_iff (b : β) (f : α → β) {g : β → γ} (hg : Injective g) :
    g ∘ f = Function.const _ (g b) ↔ f = Function.const _ b :=
  hg.comp_left.eq_iff' rfl


@[to_additive]
lemma comp_eq_one_iff [One β] [One γ] (f : α → β) {g : β → γ} (hg : Injective g) (hg0 : g 1 = 1) :
    g ∘ f = 1 ↔ f = 1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : One β
    inst✝ : One γ
    f : α → β
    g : β → γ
    hg : Function.Injective g
    hg0 : Eq (g 1) 1
    ⊢ Iff (Eq (Function.comp g f) 1) (Eq f 1)
  -/
  simpa [hg0, const_one] using comp_eq_const_iff 1 f hg
  /-
    🎉 no goals
  -/


@[to_additive]
lemma comp_ne_one_iff [One β] [One γ] (f : α → β) {g : β → γ} (hg : Injective g) (hg0 : g 1 = 1) :
    g ∘ f ≠ 1 ↔ f ≠ 1 :=
  (comp_eq_one_iff f hg hg0).ne


/-- If the one function is surjective, the codomain is trivial. -/
@[to_additive "If the zero function is surjective, the codomain is trivial."]
def uniqueOfSurjectiveOne (α : Type*) {β : Type*} [One β] (h : Function.Surjective (1 : α → β)) :
    Unique β :=
  h.uniqueOfSurjectiveConst α (1 : β)


@[to_additive]
theorem Subsingleton.pi_mulSingle_eq {α : Type*} [DecidableEq I] [Subsingleton I] [One α]
    (i : I) (x : α) : Pi.mulSingle i x = fun _ => x :=
                     /-
                       I : Type u
                       α : Type u_4
                       inst✝² : DecidableEq I
                       inst✝¹ : Subsingleton I
                       inst✝ : One α
                       i : I
                       x : α
                       j : I
                       ⊢ Eq (Pi.mulSingle i x j) x
                     -/
  funext fun j => by rw [Subsingleton.elim j i, Pi.mulSingle_eq_same]
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp)]
theorem elim_one_one [One γ] : Sum.elim (1 : α → γ) (1 : β → γ) = 1 :=
  Sum.elim_const_const 1


@[to_additive (attr := simp)]
theorem elim_mulSingle_one [DecidableEq α] [DecidableEq β] [One γ] (i : α) (c : γ) :
    Sum.elim (Pi.mulSingle i c) (1 : β → γ) = Pi.mulSingle (Sum.inl i) c := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : One γ
    i : α
    c : γ
    ⊢ Eq (Sum.elim (Pi.mulSingle i c) 1) (Pi.mulSingle (Sum.inl i) c)
  -/
  simp only [Pi.mulSingle, Sum.elim_update_left, elim_one_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem elim_one_mulSingle [DecidableEq α] [DecidableEq β] [One γ] (i : β) (c : γ) :
    Sum.elim (1 : α → γ) (Pi.mulSingle i c) = Pi.mulSingle (Sum.inr i) c := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : One γ
    i : β
    c : γ
    ⊢ Eq (Sum.elim 1 (Pi.mulSingle i c)) (Pi.mulSingle (Sum.inr i) c)
  -/
  simp only [Pi.mulSingle, Sum.elim_update_right, elim_one_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem elim_inv_inv [Inv γ] : Sum.elim a⁻¹ b⁻¹ = (Sum.elim a b)⁻¹ :=
  (Sum.comp_elim Inv.inv a b).symm


@[to_additive]
theorem elim_mul_mul [Mul γ] : Sum.elim (a * a') (b * b') = Sum.elim a b * Sum.elim a' b' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a a' : α → γ
    b b' : β → γ
    inst✝ : Mul γ
    ⊢ Eq (Sum.elim (HMul.hMul a a') (HMul.hMul b b')) (HMul.hMul (Sum.elim a b) (S …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a a' : α → γ
    b b' : β → γ
    inst✝ : Mul γ
    x : Sum α β
    ⊢ Eq (Sum.elim (HMul.hMul a a') (HMul.hMul b b') x) (HMul.hMul (Sum.elim a b)  …
  -/
              /-
                🎉 no goals
              -/
  cases x <;> rfl
              /-
                🎉 no goals
              -/


@[to_additive]
theorem elim_div_div [Div γ] : Sum.elim (a / a') (b / b') = Sum.elim a b / Sum.elim a' b' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a a' : α → γ
    b b' : β → γ
    inst✝ : Div γ
    ⊢ Eq (Sum.elim (HDiv.hDiv a a') (HDiv.hDiv b b')) (HDiv.hDiv (Sum.elim a b) (S …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a a' : α → γ
    b b' : β → γ
    inst✝ : Div γ
    x : Sum α β
    ⊢ Eq (Sum.elim (HDiv.hDiv a a') (HDiv.hDiv b b') x) (HDiv.hDiv (Sum.elim a b)  …
  -/
              /-
                🎉 no goals
              -/
  cases x <;> rfl
              /-
                🎉 no goals
              -/


