/--
The notation typeclass for heterogeneous additive actions.
This enables the notation `a +ᵥ b : γ` where `a : α`, `b : β`.
-/
class HVAdd (α : Type u) (β : Type v) (γ : outParam (Type w)) where
  /-- `a +ᵥ b` computes the sum of `a` and `b`.
  The meaning of this notation is type-dependent. -/
  hVAdd : α → β → γ


/--
The notation typeclass for heterogeneous scalar multiplication.
This enables the notation `a • b : γ` where `a : α`, `b : β`.

It is assumed to represent a left action in some sense.
The notation `a • b` is augmented with a macro (below) to have it elaborate as a left action.
Only the `b` argument participates in the elaboration algorithm: the algorithm uses the type of `b`
when calculating the type of the surrounding arithmetic expression
and it tries to insert coercions into `b` to get some `b'`
such that `a • b'` has the same type as `b'`.
See the module documentation near the macro for more details.
-/
class HSMul (α : Type u) (β : Type v) (γ : outParam (Type w)) where
  /-- `a • b` computes the product of `a` and `b`.
  The meaning of this notation is type-dependent, but it is intended to be used for left actions. -/
  hSMul : α → β → γ


/-- Type class for the `+ᵥ` notation. -/
class VAdd (G : Type u) (P : Type v) where
  /-- `a +ᵥ b` computes the sum of `a` and `b`. The meaning of this notation is type-dependent,
  but it is intended to be used for left actions. -/
  vadd : G → P → P


/-- Type class for the `-ᵥ` notation. -/
class VSub (G : outParam Type*) (P : Type*) where
  /-- `a -ᵥ b` computes the difference of `a` and `b`. The meaning of this notation is
  type-dependent, but it is intended to be used for additive torsors. -/
  vsub : P → P → G


/-- Typeclass for types with a scalar multiplication operation, denoted `•` (`\bu`) -/
@[to_additive (attr := ext)]
class SMul (M : Type u) (α : Type v) where
  /-- `a • b` computes the product of `a` and `b`. The meaning of this notation is type-dependent,
  but it is intended to be used for left actions. -/
  smul : M → α → α


@[inherit_doc] infixr:65 " +ᵥ " => HVAdd.hVAdd

@[inherit_doc] infixl:65 " -ᵥ " => VSub.vsub

@[inherit_doc] infixr:73 " • " => HSMul.hSMul


@[inherit_doc HSMul.hSMul]
macro_rules | `($x • $y) => `(leftact% HSMul.hSMul $x $y)


attribute [to_additive existing] Mul Div HMul instHMul HDiv instHDiv HSMul

@[to_additive (attr := default_instance)]
instance instHSMul {α β} [SMul α β] : HSMul α β β where
  hSMul := SMul.smul


@[to_additive]
theorem SMul.smul_eq_hSMul {α β} [SMul α β] : (SMul.smul : α → β → β) = HSMul.hSMul := rfl


attribute [to_additive existing (reorder := 1 2)] instHPow


/-- Class of types that have an inversion operation. -/
@[to_additive, notation_class]
class Inv (α : Type u) where
  /-- Invert an element of α. -/
  inv : α → α


@[inherit_doc]
postfix:max "⁻¹" => Inv.inv


@[to_additive]
lemma mul_dite (a : α) (b : P → α) (c : ¬ P → α) :
                                                                                /-
                                                                                  α : Type u_2
                                                                                  P : Prop
                                                                                  inst✝¹ : Decidable P
                                                                                  inst✝ : Mul α
                                                                                  a : α
                                                                                  b : P → α
                                                                                  c : Not P → α
                                                                                  ⊢ Eq (HMul.hMul a (dite P (fun h => b h) fun h => c h)) (dite P (fun h => HMul …
                                                                                -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    (a * if h : P then b h else c h) = if h : P then a * b h else a * c h := by split <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[to_additive]
lemma mul_ite (a b c : α) : (a * if P then b else c) = if P then a * b else a * c := mul_dite ..


@[to_additive]
lemma dite_mul (a : P → α) (b : ¬ P → α) (c : α) :
                                                                                /-
                                                                                  α : Type u_2
                                                                                  P : Prop
                                                                                  inst✝¹ : Decidable P
                                                                                  inst✝ : Mul α
                                                                                  a : P → α
                                                                                  b : Not P → α
                                                                                  c : α
                                                                                  ⊢ Eq (HMul.hMul (dite P (fun h => a h) fun h => b h) c) (dite P (fun h => HMul …
                                                                                -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    (if h : P then a h else b h) * c = if h : P then a h * c else b h * c := by split <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[to_additive]
lemma ite_mul (a b c : α) : (if P then a else b) * c = if P then a * c else b * c := dite_mul ..

-- We make `mul_ite` and `ite_mul` simp lemmas, but not `add_ite` or `ite_add`.
-- The problem we're trying to avoid is dealing with sums of the form `∑ x ∈ s, (f x + ite P 1 0)`,
-- in which `add_ite` followed by `sum_ite` would needlessly slice up
-- the `f x` terms according to whether `P` holds at `x`.
-- There doesn't appear to be a corresponding difficulty so far with `mul_ite` and `ite_mul`.

@[to_additive]
lemma dite_mul_dite (a : P → α) (b : ¬ P → α) (c : P → α) (d : ¬ P → α) :
    ((if h : P then a h else b h) * if h : P then c h else d h) =
                                                   /-
                                                     α : Type u_2
                                                     P : Prop
                                                     inst✝¹ : Decidable P
                                                     inst✝ : Mul α
                                                     a : P → α
                                                     b : Not P → α
                                                     c : P → α
                                                     d : Not P → α
                                                     ⊢ Eq (HMul.hMul (dite P (fun h => a h) fun h => b h) (dite P (fun h => c h) fu …
                                                   -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      if h : P then a h * c h else b h * d h := by split <;> rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
lemma ite_mul_ite (a b c d : α) :
                                                                                   /-
                                                                                     α : Type u_2
                                                                                     P : Prop
                                                                                     inst✝¹ : Decidable P
                                                                                     inst✝ : Mul α
                                                                                     a b c d : α
                                                                                     ⊢ Eq (HMul.hMul (ite P a b) (ite P c d)) (ite P (HMul.hMul a c) (HMul.hMul b d))
                                                                                   -/
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
    ((if P then a else b) * if P then c else d) = if P then a * c else b * d := by split <;> rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[to_additive]
lemma div_dite (a : α) (b : P → α) (c : ¬ P → α) :
                                                                                /-
                                                                                  α : Type u_2
                                                                                  P : Prop
                                                                                  inst✝¹ : Decidable P
                                                                                  inst✝ : Div α
                                                                                  a : α
                                                                                  b : P → α
                                                                                  c : Not P → α
                                                                                  ⊢ Eq (HDiv.hDiv a (dite P (fun h => b h) fun h => c h)) (dite P (fun h => HDiv …
                                                                                -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    (a / if h : P then b h else c h) = if h : P then a / b h else a / c h := by split <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[to_additive]
lemma div_ite (a b c : α) : (a / if P then b else c) = if P then a / b else a / c := div_dite ..


@[to_additive]
lemma dite_div (a : P → α) (b : ¬ P → α) (c : α) :
                                                                                /-
                                                                                  α : Type u_2
                                                                                  P : Prop
                                                                                  inst✝¹ : Decidable P
                                                                                  inst✝ : Div α
                                                                                  a : P → α
                                                                                  b : Not P → α
                                                                                  c : α
                                                                                  ⊢ Eq (HDiv.hDiv (dite P (fun h => a h) fun h => b h) c) (dite P (fun h => HDiv …
                                                                                -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    (if h : P then a h else b h) / c = if h : P then a h / c else b h / c := by split <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[to_additive]
lemma ite_div (a b c : α) : (if P then a else b) / c = if P then a / c else b / c := dite_div ..


@[to_additive]
lemma dite_div_dite (a : P → α) (b : ¬ P → α) (c : P → α) (d : ¬ P → α) :
    ((if h : P then a h else b h) / if h : P then c h else d h) =
                                                   /-
                                                     α : Type u_2
                                                     P : Prop
                                                     inst✝¹ : Decidable P
                                                     inst✝ : Div α
                                                     a : P → α
                                                     b : Not P → α
                                                     c : P → α
                                                     d : Not P → α
                                                     ⊢ Eq (HDiv.hDiv (dite P (fun h => a h) fun h => b h) (dite P (fun h => c h) fu …
                                                   -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      if h : P then a h / c h else b h / d h := by split <;> rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
lemma ite_div_ite (a b c d : α) :
    ((if P then a else b) / if P then c else d) = if P then a / c else b / d := dite_div_dite ..


