theorem Finset.sum_sym2_filter_not_isDiag {ι α} [LinearOrder ι] [AddCommMonoid α]
    (s : Finset ι) (p : Sym2 ι → α) :
    ∑ i ∈ s.sym2 with ¬ i.IsDiag, p i = ∑ i ∈ s.offDiag with i.1 < i.2, p s(i.1, i.2) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid α
    s : Finset ι
    p : Sym2 ι → α
    ⊢ Eq ((Finset.filter (fun i => Not i.IsDiag) s.sym2).sum fun i => p i) ((Finse …
  -/
  rw [Finset.offDiag_filter_lt_eq_filter_le]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid α
    s : Finset ι
    p : Sym2 ι → α
    ⊢ Eq ((Finset.filter (fun i => Not i.IsDiag) s.sym2).sum fun i => p i) ((Finse …
  -/
  conv_rhs => rw [← Finset.sum_subtype_eq_sum_filter]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid α
    s : Finset ι
    p : Sym2 ι → α
    ⊢ Eq ((Finset.filter (fun i => Not i.IsDiag) s.sym2).sum fun i => p i) ((Finse …
  -/
  refine (Finset.sum_equiv Sym2.sortEquiv.symm ?_ ?_).symm
    /-
      case refine_1
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid α
      s : Finset ι
      p : Sym2 ι → α
      ⊢ ∀ (i : Subtype fun p => LE.le p.1 p.2), Iff (Membership.mem (Finset.subtype  …
    -/
  · rintro ⟨⟨i₁, j₁⟩, hij₁⟩
    /-
      case refine_1.mk.mk
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid α
      s : Finset ι
      p : Sym2 ι → α
      i₁ j₁ : ι
      hij₁ : LE.le { fst := i₁, snd := j₁ }.1 { fst := i₁, snd := j₁ }.2
      ⊢ Iff (Membership.mem (Finset.subtype (fun i => LE.le i.1 i.2) s.offDiag) ⟨{ f …
    -/
    simp [and_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid α
      s : Finset ι
      p : Sym2 ι → α
      ⊢ ∀ (i : Subtype fun p => LE.le p.1 p.2), Membership.mem (Finset.subtype (fun  …
    -/
  · rintro ⟨⟨i₁, j₁⟩, hij₁⟩
    /-
      case refine_2.mk.mk
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid α
      s : Finset ι
      p : Sym2 ι → α
      i₁ j₁ : ι
      hij₁ : LE.le { fst := i₁, snd := j₁ }.1 { fst := i₁, snd := j₁ }.2
      ⊢ Membership.mem (Finset.subtype (fun i => LE.le i.1 i.2) s.offDiag) ⟨{ fst := …
    -/
    simp
    /-
      🎉 no goals
    -/

