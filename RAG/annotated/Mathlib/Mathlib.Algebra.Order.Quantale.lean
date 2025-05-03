/-- An additive quantale is an additive semigroup distributing over a complete lattice. -/
class IsAddQuantale (α : Type*) [AddSemigroup α] [CompleteLattice α] where
  /-- Addition is distributive over join in a quantale -/
  protected add_sSup_distrib (x : α) (s : Set α) : x + sSup s = ⨆ y ∈ s, x + y
  /-- Addition is distributive over join in a quantale -/
  protected sSup_add_distrib (s : Set α) (y : α) : sSup s + y = ⨆ x ∈ s, x + y


/-- A quantale is a semigroup distributing over a complete lattice. -/
@[to_additive]
class IsQuantale (α : Type*) [Semigroup α] [CompleteLattice α] where
  /-- Multiplication is distributive over join in a quantale -/
  protected mul_sSup_distrib (x : α) (s : Set α) : x * sSup s = ⨆ y ∈ s, x * y
  /-- Multiplication is distributive over join in a quantale -/
  protected sSup_mul_distrib (s : Set α) (y : α) : sSup s * y = ⨆ x ∈ s, x * y


@[to_additive]
theorem mul_sSup_distrib : x * sSup s = ⨆ y ∈ s, x * y := IsQuantale.mul_sSup_distrib _ _


@[to_additive]
theorem sSup_mul_distrib : sSup s * x = ⨆ y ∈ s, y * x := IsQuantale.sSup_mul_distrib _ _


/-- Left- and right- residuation operators on an additive quantale are similar
to the Heyting operator on complete lattices, but for a non-commutative logic.
I.e. `x ≤ y ⇨ₗ z ↔ x + y ≤ z` or alternatively `x ⇨ₗ y = sSup { z | z + x ≤ y }`. -/
def leftAddResiduation (x y : α) := sSup {z | z + x ≤ y}


/-- Left- and right- residuation operators on an additive quantale are similar
to the Heyting operator on complete lattices, but for a non-commutative logic.
I.e. `x ≤ y ⇨ᵣ z ↔ y + x ≤ z` or alternatively `x ⇨ₗ y = sSup { z | x + z ≤ y }`." -/
def rightAddResiduation (x y : α) := sSup {z | x + z ≤ y}


@[inherit_doc]
scoped infixr:60 " ⇨ₗ " => leftAddResiduation


@[inherit_doc]
scoped infixr:60 " ⇨ᵣ " => rightAddResiduation


/-- Left- and right-residuation operators on an additive quantale are similar to the Heyting
operator on complete lattices, but for a non-commutative logic.
I.e. `x ≤ y ⇨ₗ z ↔ x * y ≤ z` or alternatively `x ⇨ₗ y = sSup { z | z * x ≤ y }`.
-/
@[to_additive existing]
def leftMulResiduation (x y : α) := sSup {z | z * x ≤ y}


/-- Left- and right- residuation operators on an additive quantale are similar to the Heyting
operator on complete lattices, but for a non-commutative logic.
I.e. `x ≤ y ⇨ᵣ z ↔ y * x ≤ z` or alternatively `x ⇨ₗ y = sSup { z | x * z ≤ y }`.
-/
@[to_additive existing]
def rightMulResiduation (x y : α) := sSup {z | x * z ≤ y}


@[inherit_doc, to_additive existing]
scoped infixr:60 " ⇨ₗ " => leftMulResiduation


@[inherit_doc, to_additive existing]
scoped infixr:60 " ⇨ᵣ " => rightMulResiduation


@[to_additive]
theorem mul_iSup_distrib : x * ⨆ i, f i = ⨆ i, x * f i := by
  /-
    α : Type u_1
    ι : Type u_2
    x : α
    f : ι → α
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    ⊢ Eq (HMul.hMul x (iSup fun i => f i)) (iSup fun i => HMul.hMul x (f i))
  -/
  rw [iSup, mul_sSup_distrib, iSup_range]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem iSup_mul_distrib : (⨆ i, f i) * x = ⨆ i, f i * x := by
  /-
    α : Type u_1
    ι : Type u_2
    x : α
    f : ι → α
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    ⊢ Eq (HMul.hMul (iSup fun i => f i) x) (iSup fun i => HMul.hMul (f i) x)
  -/
  rw [iSup, sSup_mul_distrib, iSup_range]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_sup_distrib : x * (y ⊔ z) = (x * y) ⊔ (x * z) := by
  /-
    α : Type u_1
    x y z : α
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    ⊢ Eq (HMul.hMul x (Max.max y z)) (Max.max (HMul.hMul x y) (HMul.hMul x z))
  -/
  rw [← iSup_pair, ← sSup_pair, mul_sSup_distrib]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem sup_mul_distrib : (x ⊔ y) * z = (x * z) ⊔ (y * z) := by
  /-
    α : Type u_1
    x y z : α
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    ⊢ Eq (HMul.hMul (Max.max x y) z) (Max.max (HMul.hMul x z) (HMul.hMul y z))
  -/
  rw [← (@iSup_pair _ _ _ (fun _? => _? * z) _ _), ← sSup_pair, sSup_mul_distrib]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : MulLeftMono α where
  elim := by
    /-
      α : Type u_1
      ι : Type u_2
      x y z : α
      s : Set α
      f : ι → α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      ⊢ Covariant α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le x1 x2
    -/
    intro _ _ _; simp only; intro
    /-
      α : Type u_1
      ι : Type u_2
      x y z : α
      s : Set α
      f : ι → α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      m✝ n₁✝ n₂✝ : α
      a✝ : LE.le n₁✝ n₂✝
      ⊢ LE.le (HMul.hMul m✝ n₁✝) (HMul.hMul m✝ n₂✝)
    -/
    rwa [← left_eq_sup, ← mul_sup_distrib, sup_of_le_left]
    /-
      🎉 no goals
    -/


@[to_additive]
instance : MulRightMono α where
  elim := by
    /-
      α : Type u_1
      ι : Type u_2
      x y z : α
      s : Set α
      f : ι → α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      ⊢ Covariant α α (Function.swap fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.l …
    -/
    intro _ _ _; simp only; intro
    /-
      α : Type u_1
      ι : Type u_2
      x y z : α
      s : Set α
      f : ι → α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      m✝ n₁✝ n₂✝ : α
      a✝ : LE.le n₁✝ n₂✝
      ⊢ LE.le (Function.swap (fun x1 x2 => HMul.hMul x1 x2) m✝ n₁✝) (Function.swap ( …
    -/
    rwa [← left_eq_sup, ← sup_mul_distrib, sup_of_le_left]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem leftMulResiduation_le_iff_mul_le : x ≤ y ⇨ₗ z ↔ x * y ≤ z where
  mp h1 := by
    /-
      α : Type u_1
      x y z : α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      h1 : LE.le x (IsQuantale.leftMulResiduation y z)
      ⊢ LE.le (HMul.hMul x y) z
    -/
    apply le_trans (mul_le_mul_right' h1 _)
    simp_all only [leftMulResiduation, sSup_mul_distrib, Set.mem_setOf_eq,
      iSup_le_iff, implies_true]
  mpr h1 := le_sSup h1


@[to_additive]
theorem rightMulResiduation_le_iff_mul_le : x ≤ y ⇨ᵣ z ↔ y * x ≤ z where
  mp h1 := by
    /-
      α : Type u_1
      x y z : α
      inst✝² : Semigroup α
      inst✝¹ : CompleteLattice α
      inst✝ : IsQuantale α
      h1 : LE.le x (IsQuantale.rightMulResiduation y z)
      ⊢ LE.le (HMul.hMul y x) z
    -/
    apply le_trans (mul_le_mul_left' h1 _)
    simp_all only [rightMulResiduation, mul_sSup_distrib, Set.mem_setOf_eq,
      iSup_le_iff, implies_true]
  mpr h1 := le_sSup h1


@[to_additive (attr := simp)]
theorem bot_mul : ⊥ * x = ⊥ := by
  /-
    α : Type u_3
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    x : α
    ⊢ Eq (HMul.hMul Bot.bot x) Bot.bot
  -/
  rw [← sSup_empty, sSup_mul_distrib]
  /-
    α : Type u_3
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    x : α
    ⊢ Eq (iSup fun y => iSup fun h => HMul.hMul y x) (SupSet.sSup EmptyCollection. …
  -/
  simp only [Set.mem_empty_iff_false, not_false_eq_true, iSup_neg, iSup_bot, sSup_empty]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_bot : x * ⊥ = ⊥ := by
  /-
    α : Type u_3
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    x : α
    ⊢ Eq (HMul.hMul x Bot.bot) Bot.bot
  -/
  rw [← sSup_empty, mul_sSup_distrib]
  /-
    α : Type u_3
    inst✝² : Semigroup α
    inst✝¹ : CompleteLattice α
    inst✝ : IsQuantale α
    x : α
    ⊢ Eq (iSup fun y => iSup fun h => HMul.hMul x y) (SupSet.sSup EmptyCollection. …
  -/
  simp only [Set.mem_empty_iff_false, not_false_eq_true, iSup_neg, iSup_bot, sSup_empty]
  /-
    🎉 no goals
  -/


