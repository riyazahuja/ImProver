/-- If a linear endomorphism of a (finite, free) module `M` takes values in a submodule `p ⊆ M`,
then the trace of its restriction to `p` is equal to its trace on `M`. -/
lemma trace_restrict_eq_of_forall_mem [IsDomain R] [IsPrincipalIdealRing R]
    (p : Submodule R M) (f : M →ₗ[R] M)
    (hf : ∀ x, f x ∈ p) (hf' : ∀ x ∈ p, f x ∈ p := fun x _ ↦ hf x) :
    trace R p (f.restrict hf') = trace R M f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem p x)) (f.restrict hf …
  -/
  let ι := Module.Free.ChooseBasisIndex R M
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem p x)) (f.restrict hf …
  -/
  obtain ⟨n, snf : Basis.SmithNormalForm p ι n⟩ := p.smithNormalForm (Module.Free.chooseBasis R M)
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem p x)) (f.restrict hf …
  -/
  rw [trace_eq_matrix_trace R snf.bM, trace_eq_matrix_trace R snf.bN]
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    ⊢ Eq ((LinearMap.toMatrix snf.bN snf.bN) (f.restrict hf')).trace ((LinearMap.t …
  -/
  set A : Matrix (Fin n) (Fin n) R := toMatrix snf.bN snf.bN (f.restrict hf')
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    A : Matrix (Fin n) (Fin n) R := (LinearMap.toMatrix snf.bN snf.bN) (f.restrict …
    ⊢ Eq A.trace ((LinearMap.toMatrix snf.bM snf.bM) f).trace
  -/
  set B : Matrix ι ι R := toMatrix snf.bM snf.bM f
  have aux : ∀ i, B i i ≠ 0 → i ∈ Set.range snf.f := fun i hi ↦ by
    contrapose! hi; exact snf.repr_eq_zero_of_nmem_range ⟨_, (hf _)⟩ hi
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    A : Matrix (Fin n) (Fin n) R := (LinearMap.toMatrix snf.bN snf.bN) (f.restrict …
    B : Matrix ι ι R := (LinearMap.toMatrix snf.bM snf.bM) f
    aux : ∀ (i : ι), Ne (B i i) 0 → Membership.mem (Set.range ⇑snf.f) i
    ⊢ Eq A.trace B.trace
  -/
  change ∑ i, A i i = ∑ i, B i i
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    A : Matrix (Fin n) (Fin n) R := (LinearMap.toMatrix snf.bN snf.bN) (f.restrict …
    B : Matrix ι ι R := (LinearMap.toMatrix snf.bM snf.bM) f
    aux : ∀ (i : ι), Ne (B i i) 0 → Membership.mem (Set.range ⇑snf.f) i
    ⊢ Eq (Finset.univ.sum fun i => A i i) (Finset.univ.sum fun i => B i i)
  -/
  rw [← Finset.sum_filter_of_ne (p := fun j ↦ j ∈ Set.range snf.f) (by simpa using aux)]
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Finite R M
    inst✝² : Module.Free R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem p (f x)
    hf' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p (f x)) ⋯
    ι : Type u_2 := Module.Free.ChooseBasisIndex R M
    n : Nat
    snf : Basis.SmithNormalForm p ι n
    A : Matrix (Fin n) (Fin n) R := (LinearMap.toMatrix snf.bN snf.bN) (f.restrict …
    B : Matrix ι ι R := (LinearMap.toMatrix snf.bM snf.bM) f
    aux : ∀ (i : ι), Ne (B i i) 0 → Membership.mem (Set.range ⇑snf.f) i
    ⊢ Eq (Finset.univ.sum fun i => A i i) ((Finset.filter (fun x => Membership.mem …
  -/
  simp [A, B]
  /-
    🎉 no goals
  -/


