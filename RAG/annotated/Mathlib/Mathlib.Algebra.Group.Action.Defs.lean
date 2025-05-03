/-- See also `Monoid.toMulAction` and `MulZeroClass.toSMulWithZero`. -/
@[to_additive "See also `AddMonoid.toAddAction`"]
instance (priority := 910) Mul.toSMul (α : Type*) [Mul α] : SMul α α := ⟨(· * ·)⟩


@[to_additive (attr := simp)]
lemma smul_eq_mul (α : Type*) [Mul α] {a a' : α} : a • a' = a * a' := rfl


/-- Type class for additive monoid actions. -/
class AddAction (G : Type*) (P : Type*) [AddMonoid G] extends VAdd G P where
  /-- Zero is a neutral element for `+ᵥ` -/
  protected zero_vadd : ∀ p : P, (0 : G) +ᵥ p = p
  /-- Associativity of `+` and `+ᵥ` -/
  add_vadd : ∀ (g₁ g₂ : G) (p : P), (g₁ + g₂) +ᵥ p = g₁ +ᵥ g₂ +ᵥ p


/-- Typeclass for multiplicative actions by monoids. This generalizes group actions. -/
@[to_additive (attr := ext)]
class MulAction (α : Type*) (β : Type*) [Monoid α] extends SMul α β where
  /-- One is the neutral element for `•` -/
  protected one_smul : ∀ b : β, (1 : α) • b = b
  /-- Associativity of `•` and `*` -/
  mul_smul : ∀ (x y : α) (b : β), (x * y) • b = x • y • b


/-- A typeclass mixin saying that two additive actions on the same space commute. -/
class VAddCommClass (M N α : Type*) [VAdd M α] [VAdd N α] : Prop where
  /-- `+ᵥ` is left commutative -/
  vadd_comm : ∀ (m : M) (n : N) (a : α), m +ᵥ (n +ᵥ a) = n +ᵥ (m +ᵥ a)


/-- A typeclass mixin saying that two multiplicative actions on the same space commute. -/
@[to_additive]
class SMulCommClass (M N α : Type*) [SMul M α] [SMul N α] : Prop where
  /-- `•` is left commutative -/
  smul_comm : ∀ (m : M) (n : N) (a : α), m • n • a = n • m • a


/-- Commutativity of actions is a symmetric relation. This lemma can't be an instance because this
would cause a loop in the instance search graph. -/
@[to_additive]
lemma SMulCommClass.symm (M N α : Type*) [SMul M α] [SMul N α] [SMulCommClass M N α] :
    SMulCommClass N M α where smul_comm a' a b := (smul_comm a a' b).symm


@[to_additive]
lemma Function.Injective.smulCommClass [SMul M α] [SMul N α] [SMul M β] [SMul N β]
    [SMulCommClass M N β] {f : α → β} (hf : Injective f) (h₁ : ∀ (c : M) x, f (c • x) = c • f x)
    (h₂ : ∀ (c : N) x, f (c • x) = c • f x) : SMulCommClass M N α where
                                /-
                                  M : Type u_1
                                  N : Type u_2
                                  α : Type u_5
                                  β : Type u_6
                                  inst✝⁴ : SMul M α
                                  inst✝³ : SMul N α
                                  inst✝² : SMul M β
                                  inst✝¹ : SMul N β
                                  inst✝ : SMulCommClass M N β
                                  f : α → β
                                  hf : Function.Injective f
                                  h₁ : ∀ (c : M) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                  h₂ : ∀ (c : N) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                  c₁ : M
                                  c₂ : N
                                  x : α
                                  ⊢ Eq (f (HSMul.hSMul c₁ (HSMul.hSMul c₂ x))) (f (HSMul.hSMul c₂ (HSMul.hSMul c …
                                -/
  smul_comm c₁ c₂ x := hf <| by simp only [h₁, h₂, smul_comm c₁ c₂ (f x)]
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
lemma Function.Surjective.smulCommClass [SMul M α] [SMul N α] [SMul M β] [SMul N β]
    [SMulCommClass M N α] {f : α → β} (hf : Surjective f) (h₁ : ∀ (c : M) x, f (c • x) = c • f x)
    (h₂ : ∀ (c : N) x, f (c • x) = c • f x) : SMulCommClass M N β where
                                            /-
                                              M : Type u_1
                                              N : Type u_2
                                              α : Type u_5
                                              β : Type u_6
                                              inst✝⁴ : SMul M α
                                              inst✝³ : SMul N α
                                              inst✝² : SMul M β
                                              inst✝¹ : SMul N β
                                              inst✝ : SMulCommClass M N α
                                              f : α → β
                                              hf : Function.Surjective f
                                              h₁ : ∀ (c : M) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                              h₂ : ∀ (c : N) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                              c₁ : M
                                              c₂ : N
                                              x : α
                                              ⊢ Eq (HSMul.hSMul c₁ (HSMul.hSMul c₂ (f x))) (HSMul.hSMul c₂ (HSMul.hSMul c₁ ( …
                                            -/
  smul_comm c₁ c₂ := hf.forall.2 fun x ↦ by simp only [← h₁, ← h₂, smul_comm c₁ c₂ x]
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive]
instance smulCommClass_self (M α : Type*) [CommMonoid M] [MulAction M α] : SMulCommClass M M α where
                         /-
                           M✝ : Type u_1
                           N : Type u_2
                           G : Type u_3
                           H : Type u_4
                           α✝ : Type u_5
                           β : Type u_6
                           γ : Type u_7
                           δ : Type u_8
                           M : Type u_9
                           α : Type u_10
                           inst✝¹ : CommMonoid M
                           inst✝ : MulAction M α
                           a a' : M
                           b : α
                           ⊢ Eq (HSMul.hSMul a (HSMul.hSMul a' b)) (HSMul.hSMul a' (HSMul.hSMul a b))
                         -/
  smul_comm a a' b := by rw [← mul_smul, mul_comm, mul_smul]
                         /-
                           🎉 no goals
                         -/


/-- An instance of `VAddAssocClass M N α` states that the additive action of `M` on `α` is
determined by the additive actions of `M` on `N` and `N` on `α`. -/
class VAddAssocClass (M N α : Type*) [VAdd M N] [VAdd N α] [VAdd M α] : Prop where
  /-- Associativity of `+ᵥ` -/
  vadd_assoc : ∀ (x : M) (y : N) (z : α), (x +ᵥ y) +ᵥ z = x +ᵥ y +ᵥ z


/-- An instance of `IsScalarTower M N α` states that the multiplicative
action of `M` on `α` is determined by the multiplicative actions of `M` on `N`
and `N` on `α`. -/
@[to_additive VAddAssocClass] -- TODO auto-translating
class IsScalarTower (M N α : Type*) [SMul M N] [SMul N α] [SMul M α] : Prop where
  /-- Associativity of `•` -/
  smul_assoc : ∀ (x : M) (y : N) (z : α), (x • y) • z = x • y • z


@[to_additive (attr := simp)]
lemma smul_assoc {M N} [SMul M N] [SMul N α] [SMul M α] [IsScalarTower M N α] (x : M) (y : N)
    (z : α) : (x • y) • z = x • y • z := IsScalarTower.smul_assoc x y z


@[to_additive]
instance Semigroup.isScalarTower [Semigroup α] : IsScalarTower α α α := ⟨mul_assoc⟩


/-- A typeclass indicating that the right (aka `AddOpposite`) and left actions by `M` on `α` are
equal, that is that `M` acts centrally on `α`. This can be thought of as a version of commutativity
for `+ᵥ`. -/
class IsCentralVAdd (M α : Type*) [VAdd M α] [VAdd Mᵃᵒᵖ α] : Prop where
  /-- The right and left actions of `M` on `α` are equal. -/
  op_vadd_eq_vadd : ∀ (m : M) (a : α), AddOpposite.op m +ᵥ a = m +ᵥ a


/-- A typeclass indicating that the right (aka `MulOpposite`) and left actions by `M` on `α` are
equal, that is that `M` acts centrally on `α`. This can be thought of as a version of commutativity
for `•`. -/
@[to_additive]
class IsCentralScalar (M α : Type*) [SMul M α] [SMul Mᵐᵒᵖ α] : Prop where
  /-- The right and left actions of `M` on `α` are equal. -/
  op_smul_eq_smul : ∀ (m : M) (a : α), MulOpposite.op m • a = m • a


@[to_additive]
lemma IsCentralScalar.unop_smul_eq_smul {M α : Type*} [SMul M α] [SMul Mᵐᵒᵖ α]
    [IsCentralScalar M α] (m : Mᵐᵒᵖ) (a : α) : MulOpposite.unop m • a = m • a := by
  /-
    M : Type u_9
    α : Type u_10
    inst✝² : SMul M α
    inst✝¹ : SMul (MulOpposite M) α
    inst✝ : IsCentralScalar M α
    m : MulOpposite M
    a : α
    ⊢ Eq (HSMul.hSMul (MulOpposite.unop m) a) (HSMul.hSMul m a)
  -/
  induction m; exact (IsCentralScalar.op_smul_eq_smul _ a).symm
               /-
                 🎉 no goals
               -/


@[to_additive]
instance (priority := 50) SMulCommClass.op_left [SMul M α] [SMul Mᵐᵒᵖ α] [IsCentralScalar M α]
    [SMul N α] [SMulCommClass M N α] : SMulCommClass Mᵐᵒᵖ N α :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    G : Type u_3
                    H : Type u_4
                    α : Type u_5
                    β : Type u_6
                    γ : Type u_7
                    δ : Type u_8
                    inst✝⁴ : SMul M α
                    inst✝³ : SMul (MulOpposite M) α
                    inst✝² : IsCentralScalar M α
                    inst✝¹ : SMul N α
                    inst✝ : SMulCommClass M N α
                    m : MulOpposite M
                    n : N
                    a : α
                    ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n a)) (HSMul.hSMul n (HSMul.hSMul m a))
                  -/
  ⟨fun m n a ↦ by rw [← unop_smul_eq_smul m (n • a), ← unop_smul_eq_smul m a, smul_comm]⟩
                  /-
                    🎉 no goals
                  -/


@[to_additive]
instance (priority := 50) SMulCommClass.op_right [SMul M α] [SMul N α] [SMul Nᵐᵒᵖ α]
    [IsCentralScalar N α] [SMulCommClass M N α] : SMulCommClass M Nᵐᵒᵖ α :=
                  /-
                    M : Type u_1
                    N : Type u_2
                    G : Type u_3
                    H : Type u_4
                    α : Type u_5
                    β : Type u_6
                    γ : Type u_7
                    δ : Type u_8
                    inst✝⁴ : SMul M α
                    inst✝³ : SMul N α
                    inst✝² : SMul (MulOpposite N) α
                    inst✝¹ : IsCentralScalar N α
                    inst✝ : SMulCommClass M N α
                    m : M
                    n : MulOpposite N
                    a : α
                    ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n a)) (HSMul.hSMul n (HSMul.hSMul m a))
                  -/
  ⟨fun m n a ↦ by rw [← unop_smul_eq_smul n (m • a), ← unop_smul_eq_smul n a, smul_comm]⟩
                  /-
                    🎉 no goals
                  -/


@[to_additive]
instance (priority := 50) IsScalarTower.op_left [SMul M α] [SMul Mᵐᵒᵖ α] [IsCentralScalar M α]
    [SMul M N] [SMul Mᵐᵒᵖ N] [IsCentralScalar M N] [SMul N α] [IsScalarTower M N α] :
    IsScalarTower Mᵐᵒᵖ N α where
                         /-
                           M : Type u_1
                           N : Type u_2
                           G : Type u_3
                           H : Type u_4
                           α : Type u_5
                           β : Type u_6
                           γ : Type u_7
                           δ : Type u_8
                           inst✝⁷ : SMul M α
                           inst✝⁶ : SMul (MulOpposite M) α
                           inst✝⁵ : IsCentralScalar M α
                           inst✝⁴ : SMul M N
                           inst✝³ : SMul (MulOpposite M) N
                           inst✝² : IsCentralScalar M N
                           inst✝¹ : SMul N α
                           inst✝ : IsScalarTower M N α
                           m : MulOpposite M
                           n : N
                           a : α
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul m n) a) (HSMul.hSMul m (HSMul.hSMul n a))
                         -/
  smul_assoc m n a := by rw [← unop_smul_eq_smul m (n • a), ← unop_smul_eq_smul m n, smul_assoc]
                         /-
                           🎉 no goals
                         -/


@[to_additive]
instance (priority := 50) IsScalarTower.op_right [SMul M α] [SMul M N] [SMul N α]
    [SMul Nᵐᵒᵖ α] [IsCentralScalar N α] [IsScalarTower M N α] : IsScalarTower M Nᵐᵒᵖ α where
  smul_assoc m n a := by
    /-
      M : Type u_1
      N : Type u_2
      G : Type u_3
      H : Type u_4
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      δ : Type u_8
      inst✝⁵ : SMul M α
      inst✝⁴ : SMul M N
      inst✝³ : SMul N α
      inst✝² : SMul (MulOpposite N) α
      inst✝¹ : IsCentralScalar N α
      inst✝ : IsScalarTower M N α
      m : M
      n : MulOpposite N
      a : α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul m n) a) (HSMul.hSMul m (HSMul.hSMul n a))
    -/
    rw [← unop_smul_eq_smul n a, ← unop_smul_eq_smul (m • n) a, MulOpposite.unop_smul, smul_assoc]
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `SMul.comp`, `MulAction.compHom`,
`DistribMulAction.compHom`, `Module.compHom`, etc. -/
@[to_additive (attr := simp) " Auxiliary definition for `VAdd.comp`, `AddAction.compHom`, etc. "]
def comp.smul (g : N → M) (n : N) (a : α) : α := g n • a


/-- An action of `M` on `α` and a function `N → M` induces an action of `N` on `α`. -/
-- See note [reducible non-instances]
-- Since this is reducible, we make sure to go via
-- `SMul.comp.smul` to prevent typeclass inference unfolding too far
@[to_additive
"An additive action of `M` on `α` and a function `N → M` induces an additive action of `N` on `α`."]
abbrev comp (g : N → M) : SMul N α where smul := SMul.comp.smul g


/-- Given a tower of scalar actions `M → α → β`, if we use `SMul.comp`
to pull back both of `M`'s actions by a map `g : N → M`, then we obtain a new
tower of scalar actions `N → α → β`.

This cannot be an instance because it can cause infinite loops whenever the `SMul` arguments
are still metavariables. -/
@[to_additive
"Given a tower of additive actions `M → α → β`, if we use `SMul.comp` to pull back both of
`M`'s actions by a map `g : N → M`, then we obtain a new tower of scalar actions `N → α → β`.

This cannot be an instance because it can cause infinite loops whenever the `SMul` arguments
are still metavariables."]
lemma comp.isScalarTower [SMul M β] [SMul α β] [IsScalarTower M α β] (g : N → M) : by
    /-
      M : Type u_1
      N : Type u_2
      G : Type u_3
      H : Type u_4
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      δ : Type u_8
      inst✝³ : SMul M α
      inst✝² : SMul M β
      inst✝¹ : SMul α β
      inst✝ : IsScalarTower M α β
      g : N → M
      ⊢ Sort ?u.9390
    -/
    haveI := comp α g; haveI := comp β g; exact IsScalarTower N α β where
                                          /-
                                            🎉 no goals
                                          -/
  __ := comp α g
  __ := comp β g
  smul_assoc n := smul_assoc (g n)


/-- This cannot be an instance because it can cause infinite loops whenever the `SMul` arguments
are still metavariables. -/
@[to_additive
"This cannot be an instance because it can cause infinite loops whenever the `VAdd` arguments
are still metavariables."]
lemma comp.smulCommClass [SMul β α] [SMulCommClass M β α] (g : N → M) :
    haveI := comp α g
    SMulCommClass N β α where
  __ := comp α g
  smul_comm n := smul_comm (g n)


/-- This cannot be an instance because it can cause infinite loops whenever the `SMul` arguments
are still metavariables. -/
@[to_additive
"This cannot be an instance because it can cause infinite loops whenever the `VAdd` arguments
are still metavariables."]
lemma comp.smulCommClass' [SMul β α] [SMulCommClass β M α] (g : N → M) :
    haveI := comp α g
    SMulCommClass β N α where
  __ := comp α g
  smul_comm _ n := smul_comm _ (g n)


/-- Note that the `SMulCommClass α β β` typeclass argument is usually satisfied by `Algebra α β`. -/
@[to_additive] -- Porting note: nolint to_additive_doc
lemma mul_smul_comm [Mul β] [SMul α β] [SMulCommClass α β β] (s : α) (x y : β) :
    x * s • y = s • (x * y) := (smul_comm s x y).symm


/-- Note that the `IsScalarTower α β β` typeclass argument is usually satisfied by `Algebra α β`. -/
@[to_additive] -- Porting note: nolint to_additive_doc
lemma smul_mul_assoc [Mul β] [SMul α β] [IsScalarTower α β β] (r : α) (x y : β) :
    r • x * y = r • (x * y) := smul_assoc r x y


/-- Note that the `IsScalarTower α β β` typeclass argument is usually satisfied by `Algebra α β`. -/
@[to_additive]
lemma smul_div_assoc [DivInvMonoid β] [SMul α β] [IsScalarTower α β β] (r : α) (x y : β) :
                                  /-
                                    α : Type u_5
                                    β : Type u_6
                                    inst✝² : DivInvMonoid β
                                    inst✝¹ : SMul α β
                                    inst✝ : IsScalarTower α β β
                                    r : α
                                    x y : β
                                    ⊢ Eq (HDiv.hDiv (HSMul.hSMul r x) y) (HSMul.hSMul r (HDiv.hDiv x y))
                                  -/
    r • x / y = r • (x / y) := by simp [div_eq_mul_inv, smul_mul_assoc]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
lemma smul_smul_smul_comm [SMul α β] [SMul α γ] [SMul β δ] [SMul α δ] [SMul γ δ]
    [IsScalarTower α β δ] [IsScalarTower α γ δ] [SMulCommClass β γ δ] (a : α) (b : β) (c : γ)
                                                      /-
                                                        α : Type u_5
                                                        β : Type u_6
                                                        γ : Type u_7
                                                        δ : Type u_8
                                                        inst✝⁷ : SMul α β
                                                        inst✝⁶ : SMul α γ
                                                        inst✝⁵ : SMul β δ
                                                        inst✝⁴ : SMul α δ
                                                        inst✝³ : SMul γ δ
                                                        inst✝² : IsScalarTower α β δ
                                                        inst✝¹ : IsScalarTower α γ δ
                                                        inst✝ : SMulCommClass β γ δ
                                                        a : α
                                                        b : β
                                                        c : γ
                                                        d : δ
                                                        ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) (HSMul.hSMul c d)) (HSMul.hSMul (HSMul.hSM …
                                                      -/
    (d : δ) : (a • b) • c • d = (a • c) • b • d := by rw [smul_assoc, smul_assoc, smul_comm b]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Note that the `IsScalarTower α β β` and `SMulCommClass α β β` typeclass arguments are usually
satisfied by `Algebra α β`. -/
@[to_additive]
lemma smul_mul_smul_comm [Mul α] [Mul β] [SMul α β] [IsScalarTower α β β]
    [IsScalarTower α α β] [SMulCommClass α β β] (a : α) (b : β) (c : α) (d : β) :
    (a • b) * (c • d) = (a * c) • (b * d) := by
  /-
    α : Type u_5
    β : Type u_6
    inst✝⁵ : Mul α
    inst✝⁴ : Mul β
    inst✝³ : SMul α β
    inst✝² : IsScalarTower α β β
    inst✝¹ : IsScalarTower α α β
    inst✝ : SMulCommClass α β β
    a : α
    b : β
    c : α
    d : β
    ⊢ Eq (HMul.hMul (HSMul.hSMul a b) (HSMul.hSMul c d)) (HSMul.hSMul (HMul.hMul a …
  -/
  have : SMulCommClass β α β := .symm ..; exact smul_smul_smul_comm a b c d
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
alias smul_mul_smul := smul_mul_smul_comm

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

/-- Note that the `IsScalarTower α β β` and `SMulCommClass α β β` typeclass arguments are usually
satisfied by `Algebra α β`. -/
@[to_additive]
lemma mul_smul_mul_comm [Mul α] [Mul β] [SMul α β] [IsScalarTower α β β]
    [IsScalarTower α α β] [SMulCommClass α β β] (a b : α) (c d : β) :
    (a * b) • (c * d) = (a • c) * (b • d) := smul_smul_smul_comm a b c d


@[to_additive]
lemma Commute.smul_right [Mul α] [SMulCommClass M α α] [IsScalarTower M α α] {a b : α}
    (h : Commute a b) (r : M) : Commute a (r • b) :=
  (mul_smul_comm _ _ _).trans ((congr_arg _ h).trans <| (smul_mul_assoc _ _ _).symm)


@[to_additive]
lemma Commute.smul_left [Mul α] [SMulCommClass M α α] [IsScalarTower M α α] {a b : α}
    (h : Commute a b) (r : M) : Commute (r • a) b := (h.symm.smul_right r).symm


@[to_additive]
lemma smul_smul (a₁ a₂ : M) (b : α) : a₁ • a₂ • b = (a₁ * a₂) • b := (mul_smul _ _ _).symm


@[to_additive (attr := simp)]
lemma one_smul (b : α) : (1 : M) • b = b := MulAction.one_smul _


/-- `SMul` version of `one_mul_eq_id` -/
@[to_additive "`VAdd` version of `zero_add_eq_id`"]
lemma one_smul_eq_id : (((1 : M) • ·) : α → α) = id := funext <| one_smul _


/-- `SMul` version of `comp_mul_left` -/
@[to_additive "`VAdd` version of `comp_add_left`"]
lemma comp_smul_left (a₁ a₂ : M) : (a₁ • ·) ∘ (a₂ • ·) = (((a₁ * a₂) • ·) : α → α) :=
  funext fun _ ↦ (mul_smul _ _ _).symm


/-- Pullback a multiplicative action along an injective map respecting `•`.
See note [reducible non-instances]. -/
@[to_additive
    "Pullback an additive action along an injective map respecting `+ᵥ`."]
protected abbrev Function.Injective.mulAction [SMul M β] (f : β → α) (hf : Injective f)
    (smul : ∀ (c : M) (x), f (c • x) = c • f x) : MulAction M β where
  smul := (· • ·)
  one_smul x := hf <| (smul _ _).trans <| one_smul _ (f x)
                               /-
                                 M : Type u_1
                                 N : Type u_2
                                 G : Type u_3
                                 H : Type u_4
                                 α : Type u_5
                                 β : Type u_6
                                 γ : Type u_7
                                 δ : Type u_8
                                 inst✝² : Monoid M
                                 inst✝¹ : MulAction M α
                                 inst✝ : SMul M β
                                 f : β → α
                                 hf : Function.Injective f
                                 smul : ∀ (c : M) (x : β), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                 c₁ c₂ : M
                                 x : β
                                 ⊢ Eq (f (HSMul.hSMul (HMul.hMul c₁ c₂) x)) (f (HSMul.hSMul c₁ (HSMul.hSMul c₂  …
                               -/
  mul_smul c₁ c₂ x := hf <| by simp only [smul, mul_smul]
                               /-
                                 🎉 no goals
                               -/


/-- Pushforward a multiplicative action along a surjective map respecting `•`.
See note [reducible non-instances]. -/
@[to_additive
    "Pushforward an additive action along a surjective map respecting `+ᵥ`."]
protected abbrev Function.Surjective.mulAction [SMul M β] (f : α → β) (hf : Surjective f)
    (smul : ∀ (c : M) (x), f (c • x) = c • f x) : MulAction M β where
  smul := (· • ·)
                 /-
                   M : Type u_1
                   N : Type u_2
                   G : Type u_3
                   H : Type u_4
                   α : Type u_5
                   β : Type u_6
                   γ : Type u_7
                   δ : Type u_8
                   inst✝² : Monoid M
                   inst✝¹ : MulAction M α
                   inst✝ : SMul M β
                   f : α → β
                   hf : Function.Surjective f
                   smul : ∀ (c : M) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                   ⊢ ∀ (b : β), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by simp [hf.forall, ← smul]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M : Type u_1
                   N : Type u_2
                   G : Type u_3
                   H : Type u_4
                   α : Type u_5
                   β : Type u_6
                   γ : Type u_7
                   δ : Type u_8
                   inst✝² : Monoid M
                   inst✝¹ : MulAction M α
                   inst✝ : SMul M β
                   f : α → β
                   hf : Function.Surjective f
                   smul : ∀ (c : M) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                   ⊢ ∀ (x y : M) (b : β), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul x (HSMu …
                 -/
  mul_smul := by simp [hf.forall, ← smul, mul_smul]
                 /-
                   🎉 no goals
                 -/


/-- The regular action of a monoid on itself by left multiplication.

This is promoted to a module by `Semiring.toModule`. -/
-- see Note [lower instance priority]
@[to_additive
"The regular action of a monoid on itself by left addition.

This is promoted to an `AddTorsor` by `addGroup_is_addTorsor`."]
instance (priority := 910) Monoid.toMulAction : MulAction M M where
  smul := (· * ·)
  one_smul := one_mul
  mul_smul := mul_assoc


@[to_additive]
instance IsScalarTower.left : IsScalarTower M M α where
  smul_assoc x y z := mul_smul x y z


lemma smul_pow (r : M) (x : N) : ∀ n, (r • x) ^ n = r ^ n • x ^ n
            /-
              M : Type u_1
              N : Type u_2
              inst✝⁴ : Monoid M
              inst✝³ : Monoid N
              inst✝² : MulAction M N
              inst✝¹ : IsScalarTower M N N
              inst✝ : SMulCommClass M N N
              r : M
              x : N
              ⊢ Eq (HPow.hPow (HSMul.hSMul r x) 0) (HSMul.hSMul (HPow.hPow r 0) (HPow.hPow x …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  M : Type u_1
                  N : Type u_2
                  inst✝⁴ : Monoid M
                  inst✝³ : Monoid N
                  inst✝² : MulAction M N
                  inst✝¹ : IsScalarTower M N N
                  inst✝ : SMulCommClass M N N
                  r : M
                  x : N
                  n : Nat
                  ⊢ Eq (HPow.hPow (HSMul.hSMul r x) (HAdd.hAdd n 1)) (HSMul.hSMul (HPow.hPow r ( …
                -/
  | n + 1 => by rw [pow_succ', smul_pow _ _ n, smul_mul_smul_comm, ← pow_succ', ← pow_succ']
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
                                                            /-
                                                              G : Type u_3
                                                              α : Type u_5
                                                              inst✝¹ : Group G
                                                              inst✝ : MulAction G α
                                                              g : G
                                                              a : α
                                                              ⊢ Eq (HSMul.hSMul (Inv.inv g) (HSMul.hSMul g a)) a
                                                            -/
lemma inv_smul_smul (g : G) (a : α) : g⁻¹ • g • a = a := by rw [smul_smul, inv_mul_cancel, one_smul]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive (attr := simp)]
                                                            /-
                                                              G : Type u_3
                                                              α : Type u_5
                                                              inst✝¹ : Group G
                                                              inst✝ : MulAction G α
                                                              g : G
                                                              a : α
                                                              ⊢ Eq (HSMul.hSMul g (HSMul.hSMul (Inv.inv g) a)) a
                                                            -/
lemma smul_inv_smul (g : G) (a : α) : g • g⁻¹ • a = a := by rw [smul_smul, mul_inv_cancel, one_smul]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive] lemma inv_smul_eq_iff : g⁻¹ • a = b ↔ a = g • b :=
              /-
                G : Type u_3
                α : Type u_5
                inst✝¹ : Group G
                inst✝ : MulAction G α
                g : G
                a b : α
                h : Eq (HSMul.hSMul (Inv.inv g) a) b
                ⊢ Eq a (HSMul.hSMul g b)
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [← h, smul_inv_smul], fun h ↦ by rw [h, inv_smul_smul]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive] lemma eq_inv_smul_iff : a = g⁻¹ • b ↔ g • a = b :=
              /-
                G : Type u_3
                α : Type u_5
                inst✝¹ : Group G
                inst✝ : MulAction G α
                g : G
                a b : α
                h : Eq a (HSMul.hSMul (Inv.inv g) b)
                ⊢ Eq (HSMul.hSMul g a) b
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [h, smul_inv_smul], fun h ↦ by rw [← h, inv_smul_smul]⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma Commute.smul_right_iff : Commute a (g • b) ↔ Commute a b :=
  ⟨fun h ↦ inv_smul_smul g b ▸ h.smul_right g⁻¹, fun h ↦ h.smul_right g⟩


@[simp] lemma Commute.smul_left_iff : Commute (g • a) b ↔ Commute a b := by
  /-
    G : Type u_3
    H : Type u_4
    inst✝⁴ : Group G
    g : G
    inst✝³ : Mul H
    inst✝² : MulAction G H
    inst✝¹ : SMulCommClass G H H
    inst✝ : IsScalarTower G H H
    a b : H
    ⊢ Iff (Commute (HSMul.hSMul g a) b) (Commute a b)
  -/
  rw [Commute.symm_iff, Commute.smul_right_iff, Commute.symm_iff]
  /-
    🎉 no goals
  -/


lemma smul_inv (g : G) (a : H) : (g • a)⁻¹ = g⁻¹ • a⁻¹ :=
                                   /-
                                     G : Type u_3
                                     H : Type u_4
                                     inst✝⁴ : Group G
                                     inst✝³ : Group H
                                     inst✝² : MulAction G H
                                     inst✝¹ : SMulCommClass G H H
                                     inst✝ : IsScalarTower G H H
                                     g : G
                                     a : H
                                     ⊢ Eq (HMul.hMul (HSMul.hSMul g a) (HSMul.hSMul (Inv.inv g) (Inv.inv a))) 1
                                   -/
  inv_eq_of_mul_eq_one_right <| by rw [smul_mul_smul_comm, mul_inv_cancel, mul_inv_cancel, one_smul]
                                   /-
                                     🎉 no goals
                                   -/


lemma smul_zpow (g : G) (a : H) (n : ℤ) : (g • a) ^ n = g ^ n • a ^ n := by
  /-
    G : Type u_3
    H : Type u_4
    inst✝⁴ : Group G
    inst✝³ : Group H
    inst✝² : MulAction G H
    inst✝¹ : SMulCommClass G H H
    inst✝ : IsScalarTower G H H
    g : G
    a : H
    n : Int
    ⊢ Eq (HPow.hPow (HSMul.hSMul g a) n) (HSMul.hSMul (HPow.hPow g n) (HPow.hPow a …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [smul_pow, smul_inv]
              /-
                🎉 no goals
              -/


lemma SMulCommClass.of_commMonoid
    (A B G : Type*) [CommMonoid G] [SMul A G] [SMul B G]
    [IsScalarTower A G G] [IsScalarTower B G G] :
    SMulCommClass A B G where
  smul_comm r s x := by
    rw [← one_smul G (s • x), ← smul_assoc, ← one_smul G x, ← smul_assoc s 1 x,
      smul_comm, smul_assoc, one_smul, smul_assoc, one_smul]


variable (M α) in
/-- Embedding of `α` into functions `M → α` induced by a multiplicative action of `M` on `α`. -/
@[to_additive
"Embedding of `α` into functions `M → α` induced by an additive action of `M` on `α`."]
def toFun : α ↪ M → α :=
                                                                     /-
                                                                       M : Type u_1
                                                                       N : Type u_2
                                                                       G : Type u_3
                                                                       H✝ : Type u_4
                                                                       α : Type u_5
                                                                       β : Type u_6
                                                                       γ : Type u_7
                                                                       δ : Type u_8
                                                                       inst✝¹ : Monoid M
                                                                       inst✝ : MulAction M α
                                                                       y₁ y₂ : α
                                                                       H : Eq ((fun y x => HSMul.hSMul x y) y₁) ((fun y x => HSMul.hSMul x y) y₂)
                                                                       ⊢ Eq (HSMul.hSMul 1 y₁) (HSMul.hSMul 1 y₂)
                                                                     -/
  ⟨fun y x ↦ x • y, fun y₁ y₂ H ↦ one_smul M y₁ ▸ one_smul M y₂ ▸ by convert congr_fun H 1⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[to_additive (attr := simp)]
lemma toFun_apply (x : M) (y : α) : MulAction.toFun M α y x = x • y := rfl


@[to_additive]
lemma smul_one_smul {M} (N) [Monoid N] [SMul M N] [MulAction N α] [SMul M α]
    [IsScalarTower M N α] (x : M) (y : α) : (x • (1 : N)) • y = x • y := by
  /-
    α : Type u_5
    M : Type u_9
    N : Type u_10
    inst✝⁴ : Monoid N
    inst✝³ : SMul M N
    inst✝² : MulAction N α
    inst✝¹ : SMul M α
    inst✝ : IsScalarTower M N α
    x : M
    y : α
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul x 1) y) (HSMul.hSMul x y)
  -/
  rw [smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma smul_one_mul {M N} [MulOneClass N] [SMul M N] [IsScalarTower M N N] (x : M) (y : N) :
                                  /-
                                    M : Type u_9
                                    N : Type u_10
                                    inst✝² : MulOneClass N
                                    inst✝¹ : SMul M N
                                    inst✝ : IsScalarTower M N N
                                    x : M
                                    y : N
                                    ⊢ Eq (HMul.hMul (HSMul.hSMul x 1) y) (HSMul.hSMul x y)
                                  -/
    x • (1 : N) * y = x • y := by rw [smul_mul_assoc, one_mul]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive (attr := simp)]
lemma mul_smul_one {M N} [MulOneClass N] [SMul M N] [SMulCommClass M N N] (x : M) (y : N) :
                                  /-
                                    M : Type u_9
                                    N : Type u_10
                                    inst✝² : MulOneClass N
                                    inst✝¹ : SMul M N
                                    inst✝ : SMulCommClass M N N
                                    x : M
                                    y : N
                                    ⊢ Eq (HMul.hMul y (HSMul.hSMul x 1)) (HSMul.hSMul x y)
                                  -/
    y * x • (1 : N) = x • y := by rw [← smul_eq_mul, ← smul_comm, smul_eq_mul, mul_one]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
lemma IsScalarTower.of_smul_one_mul {M N} [Monoid N] [SMul M N]
    (h : ∀ (x : M) (y : N), x • (1 : N) * y = x • y) : IsScalarTower M N N :=
                  /-
                    M : Type u_9
                    N : Type u_10
                    inst✝¹ : Monoid N
                    inst✝ : SMul M N
                    h : ∀ (x : M) (y : N), Eq (HMul.hMul (HSMul.hSMul x 1) y) (HSMul.hSMul x y)
                    x : M
                    y z : N
                    ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                  -/
  ⟨fun x y z ↦ by rw [← h, smul_eq_mul, mul_assoc, h, smul_eq_mul]⟩
                  /-
                    🎉 no goals
                  -/


@[to_additive]
lemma SMulCommClass.of_mul_smul_one {M N} [Monoid N] [SMul M N]
    (H : ∀ (x : M) (y : N), y * x • (1 : N) = x • y) : SMulCommClass M N N :=
                  /-
                    M : Type u_9
                    N : Type u_10
                    inst✝¹ : Monoid N
                    inst✝ : SMul M N
                    H : ∀ (x : M) (y : N), Eq (HMul.hMul y (HSMul.hSMul x 1)) (HSMul.hSMul x y)
                    x : M
                    y z : N
                    ⊢ Eq (HSMul.hSMul x (HSMul.hSMul y z)) (HSMul.hSMul y (HSMul.hSMul x z))
                  -/
  ⟨fun x y z ↦ by rw [← H x z, smul_eq_mul, ← H, smul_eq_mul, mul_assoc]⟩
                  /-
                    🎉 no goals
                  -/


