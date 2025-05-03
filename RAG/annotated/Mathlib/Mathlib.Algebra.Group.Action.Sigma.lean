@[to_additive Sigma.VAdd]
instance : SMul M (Σi, α i) :=
  ⟨fun a => (Sigma.map id) fun _ => (a • ·)⟩


@[to_additive]
theorem smul_def : a • x = x.map id fun _ => (a • ·) :=
  rfl


@[to_additive (attr := simp)]
theorem smul_mk : a • mk i b = ⟨i, a • b⟩ :=
  rfl


@[to_additive]
instance instIsScalarTowerOfSMul [SMul M N] [∀ i, IsScalarTower M N (α i)] :
    IsScalarTower M N (Σi, α i) :=
  ⟨fun a b x => by
    /-
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝³ : (i : ι) → SMul M (α i)
      inst✝² : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b✝ : α i
      x✝ : Sigma fun i => α i
      inst✝¹ : SMul M N
      inst✝ : ∀ (i : ι), IsScalarTower M N (α i)
      a : M
      b : N
      x : Sigma fun i => α i
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) x) (HSMul.hSMul a (HSMul.hSMul b x))
    -/
    cases x
    /-
      case mk
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝³ : (i : ι) → SMul M (α i)
      inst✝² : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b✝ : α i
      x : Sigma fun i => α i
      inst✝¹ : SMul M N
      inst✝ : ∀ (i : ι), IsScalarTower M N (α i)
      a : M
      b : N
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) ⟨fst✝, snd✝⟩) (HSMul.hSMul a (HSMul.hSMul  …
    -/
    rw [smul_mk, smul_mk, smul_mk, smul_assoc]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [∀ i, SMulCommClass M N (α i)] : SMulCommClass M N (Σi, α i) :=
  ⟨fun a b x => by
    /-
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝² : (i : ι) → SMul M (α i)
      inst✝¹ : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b✝ : α i
      x✝ : Sigma fun i => α i
      inst✝ : ∀ (i : ι), SMulCommClass M N (α i)
      a : M
      b : N
      x : Sigma fun i => α i
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul b x)) (HSMul.hSMul b (HSMul.hSMul a x))
    -/
    cases x
    /-
      case mk
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝² : (i : ι) → SMul M (α i)
      inst✝¹ : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b✝ : α i
      x : Sigma fun i => α i
      inst✝ : ∀ (i : ι), SMulCommClass M N (α i)
      a : M
      b : N
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul b ⟨fst✝, snd✝⟩)) (HSMul.hSMul b (HSMul.hSMul  …
    -/
    rw [smul_mk, smul_mk, smul_mk, smul_mk, smul_comm]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [∀ i, SMul Mᵐᵒᵖ (α i)] [∀ i, IsCentralScalar M (α i)] : IsCentralScalar M (Σi, α i) :=
  ⟨fun a x => by
    /-
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝³ : (i : ι) → SMul M (α i)
      inst✝² : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b : α i
      x✝ : Sigma fun i => α i
      inst✝¹ : (i : ι) → SMul (MulOpposite M) (α i)
      inst✝ : ∀ (i : ι), IsCentralScalar M (α i)
      a : M
      x : Sigma fun i => α i
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) x) (HSMul.hSMul a x)
    -/
    cases x
    /-
      case mk
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      inst✝³ : (i : ι) → SMul M (α i)
      inst✝² : (i : ι) → SMul N (α i)
      a✝ : M
      i : ι
      b : α i
      x : Sigma fun i => α i
      inst✝¹ : (i : ι) → SMul (MulOpposite M) (α i)
      inst✝ : ∀ (i : ι), IsCentralScalar M (α i)
      a : M
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) ⟨fst✝, snd✝⟩) (HSMul.hSMul a ⟨fst✝, snd✝⟩)
    -/
    rw [smul_mk, smul_mk, op_smul_eq_smul]⟩
    /-
      🎉 no goals
    -/


/-- This is not an instance because `i` becomes a metavariable. -/
@[to_additive "This is not an instance because `i` becomes a metavariable."]
protected theorem FaithfulSMul' [FaithfulSMul M (α i)] : FaithfulSMul M (Σi, α i) :=
  ⟨fun h => eq_of_smul_eq_smul fun a : α i => heq_iff_eq.1 (Sigma.ext_iff.1 <| h <| mk i a).2⟩


@[to_additive]
instance [Nonempty ι] [∀ i, FaithfulSMul M (α i)] : FaithfulSMul M (Σi, α i) :=
  (Nonempty.elim ‹_›) fun i => Sigma.FaithfulSMul' i


@[to_additive]
instance {m : Monoid M} [∀ i, MulAction M (α i)] :
    MulAction M (Σi, α i) where
  mul_smul a b x := by
    /-
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      m : Monoid M
      inst✝ : (i : ι) → MulAction M (α i)
      a b : M
      x : Sigma fun i => α i
      ⊢ Eq (HSMul.hSMul (HMul.hMul a b) x) (HSMul.hSMul a (HSMul.hSMul b x))
    -/
    cases x
    /-
      case mk
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      m : Monoid M
      inst✝ : (i : ι) → MulAction M (α i)
      a b : M
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq (HSMul.hSMul (HMul.hMul a b) ⟨fst✝, snd✝⟩) (HSMul.hSMul a (HSMul.hSMul b  …
    -/
    /-
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      m : Monoid M
      inst✝ : (i : ι) → MulAction M (α i)
      x : Sigma fun i => α i
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    rw [smul_mk, smul_mk, smul_mk, mul_smul]
    /-
      case mk
      ι : Type u_1
      M : Type u_2
      N : Type u_3
      α : ι → Type u_4
      m : Monoid M
      inst✝ : (i : ι) → MulAction M (α i)
      fst✝ : ι
      snd✝ : α fst✝
      ⊢ Eq (HSMul.hSMul 1 ⟨fst✝, snd✝⟩) ⟨fst✝, snd✝⟩
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  one_smul x := by
    cases x
    rw [smul_mk, one_smul]


