/-- If a free module is finite, then the arbitrary basis is finite. -/
noncomputable instance Module.Free.ChooseBasisIndex.fintype (R : Type u) (M : Type v)
    [Semiring R] [AddCommMonoid M] [Module R M] [Module.Free R M] [Module.Finite R M] :
    Fintype (Module.Free.ChooseBasisIndex R M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ Fintype (Module.Free.ChooseBasisIndex R M)
  -/
  refine @Fintype.ofFinite _ ?_
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ Finite (Module.Free.ChooseBasisIndex R M)
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      M : Type v
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      h✝ : Subsingleton R
      ⊢ Finite (Module.Free.ChooseBasisIndex R M)
    -/
  · have := Module.subsingleton R M
    /-
      case inl
      R : Type u
      M : Type v
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      h✝ : Subsingleton R
      this : Subsingleton M
      ⊢ Finite (Module.Free.ChooseBasisIndex R M)
    -/
    rw [ChooseBasisIndex]
    /-
      case inl
      R : Type u
      M : Type v
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      h✝ : Subsingleton R
      this : Subsingleton M
      ⊢ Finite ↑⋯.choose
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      M : Type v
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      h✝ : Nontrivial R
      ⊢ Finite (Module.Free.ChooseBasisIndex R M)
    -/
  · exact Module.Finite.finite_basis (chooseBasis _ _)
    /-
      🎉 no goals
    -/


/-- A free module with a basis indexed by a `Fintype` is finite. -/
theorem Module.Finite.of_basis {R M ι : Type*} [Semiring R] [AddCommMonoid M] [Module R M]
    [_root_.Finite ι] (b : Basis ι R M) : Module.Finite R M := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite ι
    b : Basis ι R M
    ⊢ Module.Finite R M
  -/
  cases nonempty_fintype ι
  classical
    refine ⟨⟨Finset.univ.image b, ?_⟩⟩
    simp only [Set.image_univ, Finset.coe_univ, Finset.coe_image, Basis.span_eq]


instance Module.Finite.matrix {R ι₁ ι₂ M : Type*}
    [Semiring R] [AddCommMonoid M] [Module R M] [Module.Free R M] [Module.Finite R M]
    [_root_.Finite ι₁] [_root_.Finite ι₂] :
    Module.Finite R (Matrix ι₁ ι₂ M) := by
  /-
    R : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    M : Type u_4
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Finite ι₁
    inst✝ : Finite ι₂
    ⊢ Module.Finite R (Matrix ι₁ ι₂ M)
  -/
  cases nonempty_fintype ι₁
  /-
    case intro
    R : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    M : Type u_4
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Finite ι₁
    inst✝ : Finite ι₂
    val✝ : Fintype ι₁
    ⊢ Module.Finite R (Matrix ι₁ ι₂ M)
  -/
  cases nonempty_fintype ι₂
  /-
    case intro.intro
    R : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    M : Type u_4
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Finite ι₁
    inst✝ : Finite ι₂
    val✝¹ : Fintype ι₁
    val✝ : Fintype ι₂
    ⊢ Module.Finite R (Matrix ι₁ ι₂ M)
  -/
  exact Module.Finite.of_basis <| (Free.chooseBasis _ _).matrix _ _
  /-
    🎉 no goals
  -/


