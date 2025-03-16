/-- If `b : ι → M` and `c : κ → N` are bases then so is `fun i ↦ b i.1 ⊗ₜ c i.2 : ι × κ → M ⊗ N`. -/
def Basis.tensorProduct (b : Basis ι S M) (c : Basis κ R N) :
    Basis (ι × κ) S (M ⊗[R] N) :=
  Finsupp.basisSingleOne.map
    ((TensorProduct.AlgebraTensorModule.congr b.repr c.repr).trans <|
        (finsuppTensorFinsupp R S _ _ _ _).trans <|
          Finsupp.lcongr (Equiv.refl _) (TensorProduct.AlgebraTensorModule.rid R S S)).symm


@[simp]
theorem Basis.tensorProduct_apply (b : Basis ι S M) (c : Basis κ R N) (i : ι) (j : κ) :
    Basis.tensorProduct b c (i, j) = b i ⊗ₜ c j := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    b : Basis ι S M
    c : Basis κ R N
    i : ι
    j : κ
    ⊢ Eq ((b.tensorProduct c) { fst := i, snd := j }) (TensorProduct.tmul R (b i)  …
  -/
  simp [Basis.tensorProduct]
  /-
    🎉 no goals
  -/


theorem Basis.tensorProduct_apply' (b : Basis ι S M) (c : Basis κ R N) (i : ι × κ) :
    Basis.tensorProduct b c i = b i.1 ⊗ₜ c i.2 := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    b : Basis ι S M
    c : Basis κ R N
    i : Prod ι κ
    ⊢ Eq ((b.tensorProduct c) i) (TensorProduct.tmul R (b i.1) (c i.2))
  -/
  simp [Basis.tensorProduct]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.tensorProduct_repr_tmul_apply (b : Basis ι S M) (c : Basis κ R N) (m : M) (n : N)
    (i : ι) (j : κ) :
    (Basis.tensorProduct b c).repr (m ⊗ₜ n) (i, j) = c.repr n j • b.repr m i := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    b : Basis ι S M
    c : Basis κ R N
    m : M
    n : N
    i : ι
    j : κ
    ⊢ Eq (((b.tensorProduct c).repr (TensorProduct.tmul R m n)) { fst := i, snd := …
  -/
  simp [Basis.tensorProduct, mul_comm]
  /-
    🎉 no goals
  -/


/-- The lift of an `R`-basis of `M` to an `S`-basis of the base change `S ⊗[R] M`. -/
noncomputable
def Basis.baseChange (b : Basis ι R M) : Basis ι S (S ⊗[R] M) :=
  ((Basis.singleton Unit S).tensorProduct b).reindex (Equiv.punitProd ι)


@[simp]
lemma Basis.baseChange_repr_tmul (b : Basis ι R M) (x y i) :
    (b.baseChange S).repr (x ⊗ₜ y) i = b.repr y i • x := by
  /-
    R : Type u_1
    M : Type u_3
    ι : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_7
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    b : Basis ι R M
    x : S
    y : M
    i : ι
    ⊢ Eq (((Basis.baseChange S b).repr (TensorProduct.tmul R x y)) i) (HSMul.hSMul …
  -/
  simp [Basis.baseChange, Basis.tensorProduct]
  /-
    🎉 no goals
  -/


@[simp]
lemma Basis.baseChange_apply (b : Basis ι R M) (i) :
    b.baseChange S i = 1 ⊗ₜ b i := by
  /-
    R : Type u_1
    M : Type u_3
    ι : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_7
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    b : Basis ι R M
    i : ι
    ⊢ Eq ((Basis.baseChange S b) i) (TensorProduct.tmul R 1 (b i))
  -/
  simp [Basis.baseChange, Basis.tensorProduct]
  /-
    🎉 no goals
  -/


/--
If `{𝒞ᵢ}` is a basis for the module `N`, then every elements of `x ∈ M ⊗ N` can be uniquely written
as `∑ᵢ mᵢ ⊗ 𝒞ᵢ` for some `mᵢ ∈ M`.
-/
def TensorProduct.equivFinsuppOfBasisRight : M ⊗[R] N ≃ₗ[R] κ →₀ M :=
  LinearEquiv.lTensor M 𝒞.repr ≪≫ₗ TensorProduct.finsuppScalarRight R M κ


@[simp]
lemma TensorProduct.equivFinsuppOfBasisRight_apply_tmul (m : M) (n : N) :
    (TensorProduct.equivFinsuppOfBasisRight 𝒞) (m ⊗ₜ n) =
    (𝒞.repr n).mapRange (· • m) (zero_smul _ _) := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    κ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq κ
    𝒞 : Basis κ R N
    m : M
    n : N
    ⊢ Eq ((TensorProduct.equivFinsuppOfBasisRight 𝒞) (TensorProduct.tmul R m n)) ( …
  -/
  ext; simp [equivFinsuppOfBasisRight]
       /-
         🎉 no goals
       -/


lemma TensorProduct.equivFinsuppOfBasisRight_apply_tmul_apply
    (m : M) (n : N) (i : κ) :
    (TensorProduct.equivFinsuppOfBasisRight 𝒞) (m ⊗ₜ n) i =
    𝒞.repr n i • m := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    κ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq κ
    𝒞 : Basis κ R N
    m : M
    n : N
    i : κ
    ⊢ Eq (((TensorProduct.equivFinsuppOfBasisRight 𝒞) (TensorProduct.tmul R m n))  …
  -/
  simp only [equivFinsuppOfBasisRight_apply_tmul, Finsupp.mapRange_apply]
  /-
    🎉 no goals
  -/


lemma TensorProduct.equivFinsuppOfBasisRight_symm :
    (TensorProduct.equivFinsuppOfBasisRight 𝒞).symm.toLinearMap =
    Finsupp.lsum R fun i ↦ (TensorProduct.mk R M N).flip (𝒞 i) := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    κ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq κ
    𝒞 : Basis κ R N
    ⊢ Eq (↑(TensorProduct.equivFinsuppOfBasisRight 𝒞).symm) ((Finsupp.lsum R) fun  …
  -/
  ext; simp [equivFinsuppOfBasisRight]
       /-
         🎉 no goals
       -/


@[simp]
lemma TensorProduct.equivFinsuppOfBasisRight_symm_apply (b : κ →₀ M) :
    (TensorProduct.equivFinsuppOfBasisRight 𝒞).symm b = b.sum fun i m ↦ m ⊗ₜ 𝒞 i :=
  congr($(TensorProduct.equivFinsuppOfBasisRight_symm 𝒞) b)


omit [DecidableEq κ] in
lemma TensorProduct.sum_tmul_basis_right_injective :
    Function.Injective (Finsupp.lsum R fun i ↦ (TensorProduct.mk R M N).flip (𝒞 i)) :=
  have := Classical.decEq κ
  (equivFinsuppOfBasisRight_symm (M := M) 𝒞).symm ▸
    (TensorProduct.equivFinsuppOfBasisRight 𝒞).symm.injective


omit [DecidableEq κ] in
lemma TensorProduct.sum_tmul_basis_right_eq_zero
    (b : κ →₀ M) (h : (b.sum fun i m ↦ m ⊗ₜ[R] 𝒞 i) = 0) : b = 0 :=
  have := Classical.decEq κ
                                                                            /-
                                                                              R : Type u_1
                                                                              M : Type u_3
                                                                              N : Type u_4
                                                                              κ : Type u_6
                                                                              inst✝⁴ : CommSemiring R
                                                                              inst✝³ : AddCommMonoid M
                                                                              inst✝² : Module R M
                                                                              inst✝¹ : AddCommMonoid N
                                                                              inst✝ : Module R N
                                                                              𝒞 : Basis κ R N
                                                                              b : Finsupp κ M
                                                                              h : Eq (b.sum fun i m => TensorProduct.tmul R m (𝒞 i)) 0
                                                                              this : DecidableEq κ
                                                                              ⊢ Eq ((TensorProduct.equivFinsuppOfBasisRight 𝒞).symm b) ((TensorProduct.equiv …
                                                                            -/
  (TensorProduct.equivFinsuppOfBasisRight 𝒞).symm.injective (a₂ := 0) <| by simpa
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/--
If `{ℬᵢ}` is a basis for the module `M`, then every elements of `x ∈ M ⊗ N` can be uniquely written
as `∑ᵢ ℬᵢ ⊗ nᵢ` for some `nᵢ ∈ N`.
-/
def TensorProduct.equivFinsuppOfBasisLeft : M ⊗[R] N ≃ₗ[R] ι →₀ N :=
  TensorProduct.comm R M N ≪≫ₗ TensorProduct.equivFinsuppOfBasisRight ℬ


@[simp]
lemma TensorProduct.equivFinsuppOfBasisLeft_apply_tmul (m : M) (n : N) :
    (TensorProduct.equivFinsuppOfBasisLeft ℬ) (m ⊗ₜ n) =
    (ℬ.repr m).mapRange (· • n) (zero_smul _ _) := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq ι
    ℬ : Basis ι R M
    m : M
    n : N
    ⊢ Eq ((TensorProduct.equivFinsuppOfBasisLeft ℬ) (TensorProduct.tmul R m n)) (F …
  -/
  ext; simp [equivFinsuppOfBasisLeft]
       /-
         🎉 no goals
       -/


lemma TensorProduct.equivFinsuppOfBasisLeft_apply_tmul_apply
    (m : M) (n : N) (i : ι) :
    (TensorProduct.equivFinsuppOfBasisLeft ℬ) (m ⊗ₜ n) i =
    ℬ.repr m i • n := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq ι
    ℬ : Basis ι R M
    m : M
    n : N
    i : ι
    ⊢ Eq (((TensorProduct.equivFinsuppOfBasisLeft ℬ) (TensorProduct.tmul R m n)) i …
  -/
  simp only [equivFinsuppOfBasisLeft_apply_tmul, Finsupp.mapRange_apply]
  /-
    🎉 no goals
  -/


lemma TensorProduct.equivFinsuppOfBasisLeft_symm :
    (TensorProduct.equivFinsuppOfBasisLeft ℬ).symm.toLinearMap =
    Finsupp.lsum R fun i ↦ (TensorProduct.mk R M N) (ℬ i) := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq ι
    ℬ : Basis ι R M
    ⊢ Eq (↑(TensorProduct.equivFinsuppOfBasisLeft ℬ).symm) ((Finsupp.lsum R) fun i …
  -/
  ext; simp [equivFinsuppOfBasisLeft]
       /-
         🎉 no goals
       -/


@[simp]
lemma TensorProduct.equivFinsuppOfBasisLeft_symm_apply (b : ι →₀ N) :
    (TensorProduct.equivFinsuppOfBasisLeft ℬ).symm b = b.sum fun i n ↦ ℬ i ⊗ₜ n :=
  congr($(TensorProduct.equivFinsuppOfBasisLeft_symm ℬ) b)


omit [DecidableEq κ] in
/-- Elements in `M ⊗ N` can be represented by sum of elements in `M` tensor elements of basis of
`N`. -/
lemma TensorProduct.eq_repr_basis_right :
    ∃ b : κ →₀ M, b.sum (fun i m ↦ m ⊗ₜ 𝒞 i) = x := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    κ : Type u_6
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    𝒞 : Basis κ R N
    x : TensorProduct R M N
    ⊢ Exists fun b => Eq (b.sum fun i m => TensorProduct.tmul R m (𝒞 i)) x
  -/
  classical simpa using (TensorProduct.equivFinsuppOfBasisRight 𝒞).symm.surjective x
  /-
    🎉 no goals
  -/


omit [DecidableEq ι] in
/-- Elements in `M ⊗ N` can be represented by sum of elements of basis of `M` tensor elements of
  `N`.-/
lemma TensorProduct.eq_repr_basis_left :
    ∃ (c : ι →₀ N), (c.sum fun i n ↦ ℬ i ⊗ₜ n) = x := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ℬ : Basis ι R M
    x : TensorProduct R M N
    ⊢ Exists fun c => Eq (c.sum fun i n => TensorProduct.tmul R (ℬ i) n) x
  -/
  classical obtain ⟨c, rfl⟩ := (TensorProduct.equivFinsuppOfBasisLeft ℬ).symm.surjective x
  /-
    case intro
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ℬ : Basis ι R M
    c : Finsupp ι N
    ⊢ Exists fun c_1 => Eq (c_1.sum fun i n => TensorProduct.tmul R (ℬ i) n) ((Ten …
  -/
  exact ⟨c, (TensorProduct.comm R M N).injective <| by simp [Finsupp.sum]⟩
  /-
    🎉 no goals
  -/


omit [DecidableEq ι] in
lemma TensorProduct.sum_tmul_basis_left_injective :
    Function.Injective (Finsupp.lsum R fun i ↦ (TensorProduct.mk R M N) (ℬ i)) :=
  have := Classical.decEq ι
  (equivFinsuppOfBasisLeft_symm (N := N) ℬ).symm ▸
    (TensorProduct.equivFinsuppOfBasisLeft ℬ).symm.injective


omit [DecidableEq ι] in
lemma TensorProduct.sum_tmul_basis_left_eq_zero
    (b : ι →₀ N) (h : (b.sum fun i n ↦ ℬ i ⊗ₜ[R] n) = 0) : b = 0 :=
  have := Classical.decEq ι
                                                                           /-
                                                                             R : Type u_1
                                                                             M : Type u_3
                                                                             N : Type u_4
                                                                             ι : Type u_5
                                                                             inst✝⁴ : CommSemiring R
                                                                             inst✝³ : AddCommMonoid M
                                                                             inst✝² : Module R M
                                                                             inst✝¹ : AddCommMonoid N
                                                                             inst✝ : Module R N
                                                                             ℬ : Basis ι R M
                                                                             b : Finsupp ι N
                                                                             h : Eq (b.sum fun i n => TensorProduct.tmul R (ℬ i) n) 0
                                                                             this : DecidableEq ι
                                                                             ⊢ Eq ((TensorProduct.equivFinsuppOfBasisLeft ℬ).symm b) ((TensorProduct.equivF …
                                                                           -/
  (TensorProduct.equivFinsuppOfBasisLeft ℬ).symm.injective (a₂ := 0) <| by simpa
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


