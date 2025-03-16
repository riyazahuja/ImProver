instance Vector.fintype [Fintype α] {n : ℕ} : Fintype (List.Vector α n) :=
  Fintype.ofEquiv _ (Equiv.vectorEquivFin _ _).symm


instance [DecidableEq α] [Fintype α] {n : ℕ} : Fintype (Sym.Sym' α n) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    ⊢ Fintype (Sym.Sym' α n)
  -/
  refine @Quotient.fintype _ _ _ ?_
  -- Porting note: had to build the instance manually
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    ⊢ DecidableRel fun x1 x2 => HasEquiv.Equiv x1 x2
  -/
  intros x y
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    x y : List.Vector α n
    ⊢ Decidable ((fun x1 x2 => HasEquiv.Equiv x1 x2) x y)
  -/
  apply List.decidablePerm
  /-
    🎉 no goals
  -/


instance [DecidableEq α] [Fintype α] {n : ℕ} : Fintype (Sym α n) :=
  Fintype.ofEquiv _ Sym.symEquivSym'.symm

