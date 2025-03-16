/-- A finite module admits a surjective linear map from a finite free module. -/
lemma exists_fin' [Module.Finite R M] : ∃ (n : ℕ) (f : (Fin n → R) →ₗ[R] M), Surjective f := by
  /-
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  have ⟨n, s, hs⟩ := exists_fin (R := R) (M := M)
  /-
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    n : Nat
    s : Fin n → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  refine ⟨n, Basis.constr (Pi.basisFun R _) ℕ s, ?_⟩
  /-
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    n : Nat
    s : Fin n → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    ⊢ Function.Surjective ⇑(((Pi.basisFun R (Fin n)).constr Nat) s)
  -/
  rw [← LinearMap.range_eq_top, Basis.constr_range, hs]
  /-
    🎉 no goals
  -/


lemma small [Module.Finite R M] [Small.{v} R] : Small.{v} M :=
  have ⟨_, _, h⟩ := exists_fin' R M
  small_of_surjective h


lemma _root_.Module.finite_of_finite [Finite R] [Module.Finite R M] : Finite M := by
  /-
    R : Type u
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Finite R
    inst✝ : Module.Finite R M
    ⊢ Finite M
  -/
  obtain ⟨n, f, hf⟩ := exists_fin' R M; exact .of_surjective f hf
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-10-13")]
alias _root_.FiniteDimensional.finite_of_finite := finite_of_finite


/-- A finite dimensional vector space over a finite field is finite -/
@[deprecated (since := "2024-10-22")]
alias _root_.FiniteDimensional.fintypeOfFintype := finite_of_finite


/-- A module over a finite ring has finite dimension iff it is finite. -/
lemma _root_.Module.finite_iff_finite [Finite R] : Module.Finite R M ↔ Finite M :=
  ⟨fun _ ↦ finite_of_finite R, fun _ ↦ .of_finite⟩


variable (R) in
lemma _root_.Set.Finite.submoduleSpan [Finite R] {s : Set M} (hs : s.Finite) :
    (Submodule.span R s : Set M).Finite := by
  /-
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite R
    s : Set M
    hs : s.Finite
    ⊢ (↑(Submodule.span R s)).Finite
  -/
  lift s to Finset M using hs
  /-
    case intro
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite R
    s : Finset M
    ⊢ (↑(Submodule.span R ↑s)).Finite
  -/
  rw [Set.Finite, ← Module.finite_iff_finite (R := R)]
  /-
    case intro
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite R
    s : Finset M
    ⊢ Module.Finite R ↑↑(Submodule.span R ↑s)
  -/
  dsimp
  /-
    case intro
    R : Type u
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite R
    s : Finset M
    ⊢ Module.Finite R (Subtype fun x => Membership.mem (Submodule.span R ↑s) x)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If a free module is finite, then any arbitrary basis is finite. -/
lemma finite_basis [Nontrivial R] {ι} [Module.Finite R M]
    (b : Basis ι R M) :
    _root_.Finite ι :=
  let ⟨s, hs⟩ := ‹Module.Finite R M›
  basis_finite_of_finite_spans (↑s) s.finite_toSet hs b


lemma not_finite_of_infinite_basis [Nontrivial R] {ι} [Infinite ι] (b : Basis ι R M) :
    ¬ Module.Finite R M :=
  fun _ ↦ (Finite.finite_basis b).not_infinite ‹_›


