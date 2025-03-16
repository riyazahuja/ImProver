theorem smul_mono_right [SMul M α] [Preorder α] [CovariantClass M α HSMul.hSMul LE.le]
    (m : M) : Monotone (HSMul.hSMul m : α → α) :=
  fun _ _ => CovariantClass.elim _


/-- A copy of `smul_mono_right` that is understood by `gcongr`. -/
@[gcongr]
theorem smul_le_smul_left [SMul M α] [Preorder α] [CovariantClass M α HSMul.hSMul LE.le]
    (m : M) {a b : α} (h : a ≤ b) :
    m • a ≤ m • b :=
  smul_mono_right _ h


theorem smul_inf_le [SMul M α] [SemilatticeInf α] [CovariantClass M α HSMul.hSMul LE.le]
    (m : M) (a₁ a₂ : α) : m • (a₁ ⊓ a₂) ≤ m • a₁ ⊓ m • a₂ :=
  (smul_mono_right _).map_inf_le _ _


theorem smul_iInf_le [SMul M α] [CompleteLattice α] [CovariantClass M α HSMul.hSMul LE.le]
    {m : M} {t : ι → α} :
    m • iInf t ≤ ⨅ i, m • t i :=
  le_iInf fun _ => smul_mono_right _ (iInf_le _ _)


theorem smul_strictMono_right [SMul M α] [Preorder α] [CovariantClass M α HSMul.hSMul LT.lt]
    (m : M) : StrictMono (HSMul.hSMul m : α → α) :=
  fun _ _ => CovariantClass.elim _


lemma le_pow_smul {G : Type*} [Monoid G] {α : Type*} [Preorder α] {g : G} {a : α}
    [MulAction G α] [CovariantClass G α HSMul.hSMul LE.le]
    (h : a ≤ g • a) (n : ℕ) : a ≤ g ^ n • a := by
  /-
    G : Type u_4
    inst✝³ : Monoid G
    α : Type u_5
    inst✝² : Preorder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le a (HSMul.hSMul g a)
    n : Nat
    ⊢ LE.le a (HSMul.hSMul (HPow.hPow g n) a)
  -/
  induction' n with n hn
    /-
      case zero
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le a (HSMul.hSMul g a)
      ⊢ LE.le a (HSMul.hSMul (HPow.hPow g 0) a)
    -/
  · rw [pow_zero, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le a (HSMul.hSMul g a)
      n : Nat
      hn : LE.le a (HSMul.hSMul (HPow.hPow g n) a)
      ⊢ LE.le a (HSMul.hSMul (HPow.hPow g (HAdd.hAdd n 1)) a)
    -/
  · rw [pow_succ', mul_smul]
    /-
      case succ
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le a (HSMul.hSMul g a)
      n : Nat
      hn : LE.le a (HSMul.hSMul (HPow.hPow g n) a)
      ⊢ LE.le a (HSMul.hSMul g (HSMul.hSMul (HPow.hPow g n) a))
    -/
    exact h.trans (smul_mono_right g hn)
    /-
      🎉 no goals
    -/


lemma pow_smul_le {G : Type*} [Monoid G] {α : Type*} [Preorder α] {g : G} {a : α}
    [MulAction G α] [CovariantClass G α HSMul.hSMul LE.le]
    (h : g • a ≤ a) (n : ℕ) : g ^ n • a ≤ a := by
  /-
    G : Type u_4
    inst✝³ : Monoid G
    α : Type u_5
    inst✝² : Preorder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le (HSMul.hSMul g a) a
    n : Nat
    ⊢ LE.le (HSMul.hSMul (HPow.hPow g n) a) a
  -/
  induction' n with n hn
    /-
      case zero
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le (HSMul.hSMul g a) a
      ⊢ LE.le (HSMul.hSMul (HPow.hPow g 0) a) a
    -/
  · rw [pow_zero, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le (HSMul.hSMul g a) a
      n : Nat
      hn : LE.le (HSMul.hSMul (HPow.hPow g n) a) a
      ⊢ LE.le (HSMul.hSMul (HPow.hPow g (HAdd.hAdd n 1)) a) a
    -/
  · rw [pow_succ', mul_smul]
    /-
      case succ
      G : Type u_4
      inst✝³ : Monoid G
      α : Type u_5
      inst✝² : Preorder α
      g : G
      a : α
      inst✝¹ : MulAction G α
      inst✝ : CovariantClass G α HSMul.hSMul LE.le
      h : LE.le (HSMul.hSMul g a) a
      n : Nat
      hn : LE.le (HSMul.hSMul (HPow.hPow g n) a) a
      ⊢ LE.le (HSMul.hSMul g (HSMul.hSMul (HPow.hPow g n) a)) a
    -/
    exact (smul_mono_right g hn).trans h
    /-
      🎉 no goals
    -/

