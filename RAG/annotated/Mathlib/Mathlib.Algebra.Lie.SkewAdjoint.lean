theorem LinearMap.BilinForm.isSkewAdjoint_bracket {f g : Module.End R M}
    (hf : f ∈ B.skewAdjointSubmodule) (hg : g ∈ B.skewAdjointSubmodule) :
    ⁅f, g⁆ ∈ B.skewAdjointSubmodule := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    f g : Module.End R M
    hf : Membership.mem (LinearMap.skewAdjointSubmodule B) f
    hg : Membership.mem (LinearMap.skewAdjointSubmodule B) g
    ⊢ Membership.mem (LinearMap.skewAdjointSubmodule B) (Bracket.bracket f g)
  -/
  rw [mem_skewAdjointSubmodule] at *
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    f g : Module.End R M
    hf : LinearMap.IsSkewAdjoint B ⇑f
    hg : LinearMap.IsSkewAdjoint B ⇑g
    ⊢ LinearMap.IsSkewAdjoint B ⇑(Bracket.bracket f g)
  -/
  have hfg : IsAdjointPair B B (f * g) (g * f) := by rw [← neg_mul_neg g f]; exact hg.comp hf
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    f g : Module.End R M
    hf : LinearMap.IsSkewAdjoint B ⇑f
    hg : LinearMap.IsSkewAdjoint B ⇑g
    hfg : LinearMap.IsAdjointPair B B ⇑(HMul.hMul f g) ⇑(HMul.hMul g f)
    ⊢ LinearMap.IsSkewAdjoint B ⇑(Bracket.bracket f g)
  -/
  have hgf : IsAdjointPair B B (g * f) (f * g) := by rw [← neg_mul_neg f g]; exact hf.comp hg
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    f g : Module.End R M
    hf : LinearMap.IsSkewAdjoint B ⇑f
    hg : LinearMap.IsSkewAdjoint B ⇑g
    hfg : LinearMap.IsAdjointPair B B ⇑(HMul.hMul f g) ⇑(HMul.hMul g f)
    hgf : LinearMap.IsAdjointPair B B ⇑(HMul.hMul g f) ⇑(HMul.hMul f g)
    ⊢ LinearMap.IsSkewAdjoint B ⇑(Bracket.bracket f g)
  -/
  change IsAdjointPair B B (f * g - g * f) (-(f * g - g * f)); rw [neg_sub]
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    f g : Module.End R M
    hf : LinearMap.IsSkewAdjoint B ⇑f
    hg : LinearMap.IsSkewAdjoint B ⇑g
    hfg : LinearMap.IsAdjointPair B B ⇑(HMul.hMul f g) ⇑(HMul.hMul g f)
    hgf : LinearMap.IsAdjointPair B B ⇑(HMul.hMul g f) ⇑(HMul.hMul f g)
    ⊢ LinearMap.IsAdjointPair B B ⇑(HSub.hSub (HMul.hMul f g) (HMul.hMul g f)) ⇑(H …
  -/
  exact hfg.sub hgf
  /-
    🎉 no goals
  -/


/-- Given an `R`-module `M`, equipped with a bilinear form, the skew-adjoint endomorphisms form a
Lie subalgebra of the Lie algebra of endomorphisms. -/
def skewAdjointLieSubalgebra : LieSubalgebra R (Module.End R M) :=
  { B.skewAdjointSubmodule with
    lie_mem' := B.isSkewAdjoint_bracket }


/-- An equivalence of modules with bilinear forms gives equivalence of Lie algebras of skew-adjoint
endomorphisms. -/
def skewAdjointLieSubalgebraEquiv :
    skewAdjointLieSubalgebra (B.compl₁₂ (e : N →ₗ[R] M) e) ≃ₗ⁅R⁆ skewAdjointLieSubalgebra B := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) N M
    ⊢ LieEquiv R (Subtype fun x => Membership.mem (skewAdjointLieSubalgebra (Linea …
  -/
  apply LieEquiv.ofSubalgebras _ _ e.lieConj
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) N M
    ⊢ Eq (LieSubalgebra.map e.lieConj.toLieHom (skewAdjointLieSubalgebra (LinearMa …
  -/
  ext f
  simp only [LieSubalgebra.mem_coe, Submodule.mem_map_equiv, LieSubalgebra.mem_map_submodule,
    LinearEquiv.coe_coe]
  /-
    case h
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) N M
    f : Module.End R M
    ⊢ Iff (Membership.mem (skewAdjointLieSubalgebra (LinearMap.compl₁₂ B ↑e ↑e)).t …
  -/
  exact (LinearMap.isPairSelfAdjoint_equiv (B := -B) (F := B) e f).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem skewAdjointLieSubalgebraEquiv_apply
    (f : skewAdjointLieSubalgebra (B.compl₁₂ (Qₗ := N) (Qₗ' := N) ↑e ↑e)) :
    ↑(skewAdjointLieSubalgebraEquiv B e f) = e.lieConj f := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) N M
    f : Subtype fun x => Membership.mem (skewAdjointLieSubalgebra (LinearMap.compl …
    ⊢ Eq (↑((skewAdjointLieSubalgebraEquiv B e) f)) (e.lieConj ↑f)
  -/
  simp [skewAdjointLieSubalgebraEquiv]
  /-
    🎉 no goals
  -/


@[simp]
theorem skewAdjointLieSubalgebraEquiv_symm_apply (f : skewAdjointLieSubalgebra B) :
    ↑((skewAdjointLieSubalgebraEquiv B e).symm f) = e.symm.lieConj f := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) N M
    f : Subtype fun x => Membership.mem (skewAdjointLieSubalgebra B) x
    ⊢ Eq (↑((skewAdjointLieSubalgebraEquiv B e).symm f)) (e.symm.lieConj ↑f)
  -/
  simp [skewAdjointLieSubalgebraEquiv]
  /-
    🎉 no goals
  -/


theorem Matrix.lie_transpose (A B : Matrix n n R) : ⁅A, B⁆ᵀ = ⁅Bᵀ, Aᵀ⁆ :=
                                               /-
                                                 R : Type u
                                                 n : Type w
                                                 inst✝² : CommRing R
                                                 inst✝¹ : DecidableEq n
                                                 inst✝ : Fintype n
                                                 A B : Matrix n n R
                                                 ⊢ Eq (HSub.hSub (HMul.hMul A B) (HMul.hMul B A)).transpose (HSub.hSub (HMul.hM …
                                               -/
  show (A * B - B * A)ᵀ = Bᵀ * Aᵀ - Aᵀ * Bᵀ by simp
                                               /-
                                                 🎉 no goals
                                               -/

-- Porting note: Changed `(A B)` to `{A B}` for convenience in `skewAdjointMatricesLieSubalgebra`

theorem Matrix.isSkewAdjoint_bracket {A B : Matrix n n R} (hA : A ∈ skewAdjointMatricesSubmodule J)
    (hB : B ∈ skewAdjointMatricesSubmodule J) : ⁅A, B⁆ ∈ skewAdjointMatricesSubmodule J := by
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J A B : Matrix n n R
    hA : Membership.mem (skewAdjointMatricesSubmodule J) A
    hB : Membership.mem (skewAdjointMatricesSubmodule J) B
    ⊢ Membership.mem (skewAdjointMatricesSubmodule J) (Bracket.bracket A B)
  -/
  simp only [mem_skewAdjointMatricesSubmodule] at *
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J A B : Matrix n n R
    hA : J.IsSkewAdjoint A
    hB : J.IsSkewAdjoint B
    ⊢ J.IsSkewAdjoint (Bracket.bracket A B)
  -/
  change ⁅A, B⁆ᵀ * J = J * (-⁅A, B⁆)
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J A B : Matrix n n R
    hA : J.IsSkewAdjoint A
    hB : J.IsSkewAdjoint B
    ⊢ Eq (HMul.hMul (Bracket.bracket A B).transpose J) (HMul.hMul J (Neg.neg (Brac …
  -/
  change Aᵀ * J = J * (-A) at hA
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J A B : Matrix n n R
    hB : J.IsSkewAdjoint B
    hA : Eq (HMul.hMul A.transpose J) (HMul.hMul J (Neg.neg A))
    ⊢ Eq (HMul.hMul (Bracket.bracket A B).transpose J) (HMul.hMul J (Neg.neg (Brac …
  -/
  change Bᵀ * J = J * (-B) at hB
  rw [Matrix.lie_transpose, LieRing.of_associative_ring_bracket,
    LieRing.of_associative_ring_bracket, sub_mul, mul_assoc, mul_assoc, hA, hB, ← mul_assoc,
    ← mul_assoc, hA, hB]
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J A B : Matrix n n R
    hA : Eq (HMul.hMul A.transpose J) (HMul.hMul J (Neg.neg A))
    hB : Eq (HMul.hMul B.transpose J) (HMul.hMul J (Neg.neg B))
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul J (Neg.neg B)) (Neg.neg A)) (HMul.hMul ( …
  -/
  noncomm_ring
  /-
    🎉 no goals
  -/


/-- The Lie subalgebra of skew-adjoint square matrices corresponding to a square matrix `J`. -/
def skewAdjointMatricesLieSubalgebra : LieSubalgebra R (Matrix n n R) :=
  { skewAdjointMatricesSubmodule J with
    lie_mem' := J.isSkewAdjoint_bracket }


@[simp]
theorem mem_skewAdjointMatricesLieSubalgebra (A : Matrix n n R) :
    A ∈ skewAdjointMatricesLieSubalgebra J ↔ A ∈ skewAdjointMatricesSubmodule J :=
  Iff.rfl


/-- An invertible matrix `P` gives a Lie algebra equivalence between those endomorphisms that are
skew-adjoint with respect to a square matrix `J` and those with respect to `PᵀJP`. -/
def skewAdjointMatricesLieSubalgebraEquiv (P : Matrix n n R) (h : Invertible P) :
    skewAdjointMatricesLieSubalgebra J ≃ₗ⁅R⁆ skewAdjointMatricesLieSubalgebra (Pᵀ * J * P) :=
  LieEquiv.ofSubalgebras _ _ (P.lieConj h).symm <| by
    /-
      R : Type u
      n : Type w
      inst✝² : CommRing R
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      J P : Matrix n n R
      h : Invertible P
      ⊢ Eq (LieSubalgebra.map (P.lieConj h).symm.toLieHom (skewAdjointMatricesLieSub …
    -/
    ext A
    suffices P.lieConj h A ∈ skewAdjointMatricesSubmodule J ↔
        A ∈ skewAdjointMatricesSubmodule (Pᵀ * J * P) by
      simp only [LieSubalgebra.mem_coe, Submodule.mem_map_equiv, LieSubalgebra.mem_map_submodule,
        LinearEquiv.coe_coe]
      exact this
    /-
      case h
      R : Type u
      n : Type w
      inst✝² : CommRing R
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      J P : Matrix n n R
      h : Invertible P
      A : Matrix n n R
      ⊢ Iff (Membership.mem (skewAdjointMatricesSubmodule J) ((P.lieConj h) A)) (Mem …
    -/
    simp [Matrix.IsSkewAdjoint, J.isAdjointPair_equiv _ _ P (isUnit_of_invertible P)]
    /-
      🎉 no goals
    -/


theorem skewAdjointMatricesLieSubalgebraEquiv_apply (P : Matrix n n R) (h : Invertible P)
    (A : skewAdjointMatricesLieSubalgebra J) :
    ↑(skewAdjointMatricesLieSubalgebraEquiv J P h A) = P⁻¹ * A * P := by
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    J P : Matrix n n R
    h : Invertible P
    A : Subtype fun x => Membership.mem (skewAdjointMatricesLieSubalgebra J) x
    ⊢ Eq (↑((skewAdjointMatricesLieSubalgebraEquiv J P h) A)) (HMul.hMul (HMul.hMu …
  -/
  simp [skewAdjointMatricesLieSubalgebraEquiv]
  /-
    🎉 no goals
  -/


/-- An equivalence of matrix algebras commuting with the transpose endomorphisms restricts to an
equivalence of Lie algebras of skew-adjoint matrices. -/
def skewAdjointMatricesLieSubalgebraEquivTranspose {m : Type w} [DecidableEq m] [Fintype m]
    (e : Matrix n n R ≃ₐ[R] Matrix m m R) (h : ∀ A, (e A)ᵀ = e Aᵀ) :
    skewAdjointMatricesLieSubalgebra J ≃ₗ⁅R⁆ skewAdjointMatricesLieSubalgebra (e J) :=
  LieEquiv.ofSubalgebras _ _ e.toLieEquiv <| by
    /-
      R : Type u
      n : Type w
      inst✝⁴ : CommRing R
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      J : Matrix n n R
      m : Type w
      inst✝¹ : DecidableEq m
      inst✝ : Fintype m
      e : AlgEquiv R (Matrix n n R) (Matrix m m R)
      h : ∀ (A : Matrix n n R), Eq (e A).transpose (e A.transpose)
      ⊢ Eq (LieSubalgebra.map e.toLieEquiv.toLieHom (skewAdjointMatricesLieSubalgebr …
    -/
    ext A
    suffices J.IsSkewAdjoint (e.symm A) ↔ (e J).IsSkewAdjoint A by
      -- Porting note: Originally `simpa [this]`
      simpa [- LieSubalgebra.mem_map, LieSubalgebra.mem_map_submodule]
    simp only [Matrix.IsSkewAdjoint, Matrix.IsAdjointPair, ← h,
      ← Function.Injective.eq_iff e.injective, map_mul, AlgEquiv.apply_symm_apply, map_neg]


@[simp]
theorem skewAdjointMatricesLieSubalgebraEquivTranspose_apply {m : Type w} [DecidableEq m]
    [Fintype m] (e : Matrix n n R ≃ₐ[R] Matrix m m R) (h : ∀ A, (e A)ᵀ = e Aᵀ)
    (A : skewAdjointMatricesLieSubalgebra J) :
    (skewAdjointMatricesLieSubalgebraEquivTranspose J e h A : Matrix m m R) = e A :=
  rfl


theorem mem_skewAdjointMatricesLieSubalgebra_unit_smul (u : Rˣ) (J A : Matrix n n R) :
    A ∈ skewAdjointMatricesLieSubalgebra (u • J) ↔ A ∈ skewAdjointMatricesLieSubalgebra J := by
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    u : Units R
    J A : Matrix n n R
    ⊢ Iff (Membership.mem (skewAdjointMatricesLieSubalgebra (HSMul.hSMul u J)) A)  …
  -/
  change A ∈ skewAdjointMatricesSubmodule (u • J) ↔ A ∈ skewAdjointMatricesSubmodule J
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    u : Units R
    J A : Matrix n n R
    ⊢ Iff (Membership.mem (skewAdjointMatricesSubmodule (HSMul.hSMul u J)) A) (Mem …
  -/
  simp only [mem_skewAdjointMatricesSubmodule, Matrix.IsSkewAdjoint, Matrix.IsAdjointPair]
  /-
    R : Type u
    n : Type w
    inst✝² : CommRing R
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    u : Units R
    J A : Matrix n n R
    ⊢ Iff (Eq (HMul.hMul A.transpose (HSMul.hSMul u J)) (HMul.hMul (HSMul.hSMul u  …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u
      n : Type w
      inst✝² : CommRing R
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      u : Units R
      J A : Matrix n n R
      h : Eq (HMul.hMul A.transpose (HSMul.hSMul u J)) (HMul.hMul (HSMul.hSMul u J)  …
      ⊢ Eq (HMul.hMul A.transpose J) (HMul.hMul J (Neg.neg A))
    -/
  · simpa using congr_arg (fun B => u⁻¹ • B) h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      n : Type w
      inst✝² : CommRing R
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      u : Units R
      J A : Matrix n n R
      h : Eq (HMul.hMul A.transpose J) (HMul.hMul J (Neg.neg A))
      ⊢ Eq (HMul.hMul A.transpose (HSMul.hSMul u J)) (HMul.hMul (HSMul.hSMul u J) (N …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


