instance [TopologicalSpace R] : TopologicalSpace (Matrix m n R) :=
  Pi.topologicalSpace


instance [TopologicalSpace R] [T2Space R] : T2Space (Matrix m n R) :=
  Pi.t2Space


instance [SMul α R] [ContinuousConstSMul α R] : ContinuousConstSMul α (Matrix m n R) :=
  inferInstanceAs (ContinuousConstSMul α (m → n → R))


instance [TopologicalSpace α] [SMul α R] [ContinuousSMul α R] : ContinuousSMul α (Matrix m n R) :=
  inferInstanceAs (ContinuousSMul α (m → n → R))


instance [Add R] [ContinuousAdd R] : ContinuousAdd (Matrix m n R) :=
  Pi.continuousAdd


instance [Neg R] [ContinuousNeg R] : ContinuousNeg (Matrix m n R) :=
  Pi.continuousNeg


instance [AddGroup R] [TopologicalAddGroup R] : TopologicalAddGroup (Matrix m n R) :=
  Pi.topologicalAddGroup


/-- To show a function into matrices is continuous it suffices to show the coefficients of the
resulting matrix are continuous -/
@[continuity]
theorem continuous_matrix [TopologicalSpace α] {f : α → Matrix m n R}
    (h : ∀ i j, Continuous fun a => f a i j) : Continuous f :=
  continuous_pi fun _ => continuous_pi fun _ => h _ _


theorem Continuous.matrix_elem {A : X → Matrix m n R} (hA : Continuous A) (i : m) (j : n) :
    Continuous fun x => A x i j :=
  (continuous_apply_apply i j).comp hA


@[continuity]
theorem Continuous.matrix_map [TopologicalSpace S] {A : X → Matrix m n S} {f : S → R}
    (hA : Continuous A) (hf : Continuous f) : Continuous fun x => (A x).map f :=
  continuous_matrix fun _ _ => hf.comp <| hA.matrix_elem _ _


@[continuity]
theorem Continuous.matrix_transpose {A : X → Matrix m n R} (hA : Continuous A) :
    Continuous fun x => (A x)ᵀ :=
  continuous_matrix fun i j => hA.matrix_elem j i


theorem Continuous.matrix_conjTranspose [Star R] [ContinuousStar R] {A : X → Matrix m n R}
    (hA : Continuous A) : Continuous fun x => (A x)ᴴ :=
  hA.matrix_transpose.matrix_map continuous_star


instance [Star R] [ContinuousStar R] : ContinuousStar (Matrix m m R) :=
  ⟨continuous_id.matrix_conjTranspose⟩


@[continuity]
theorem Continuous.matrix_col {ι : Type*} {A : X → n → R} (hA : Continuous A) :
    Continuous fun x => col ι (A x) :=
  continuous_matrix fun i _ => (continuous_apply i).comp hA


@[continuity]
theorem Continuous.matrix_row {ι : Type*} {A : X → n → R} (hA : Continuous A) :
    Continuous fun x => row ι (A x) :=
  continuous_matrix fun _ _ => (continuous_apply _).comp hA


@[continuity]
theorem Continuous.matrix_diagonal [Zero R] [DecidableEq n] {A : X → n → R} (hA : Continuous A) :
    Continuous fun x => diagonal (A x) :=
  continuous_matrix fun i _ => ((continuous_apply i).comp hA).if_const _ continuous_zero


@[continuity]
theorem Continuous.matrix_dotProduct [Fintype n] [Mul R] [AddCommMonoid R] [ContinuousAdd R]
    [ContinuousMul R] {A : X → n → R} {B : X → n → R} (hA : Continuous A) (hB : Continuous B) :
    Continuous fun x => dotProduct (A x) (B x) :=
  continuous_finset_sum _ fun i _ =>
    ((continuous_apply i).comp hA).mul ((continuous_apply i).comp hB)


/-- For square matrices the usual `continuous_mul` can be used. -/
@[continuity]
theorem Continuous.matrix_mul [Fintype n] [Mul R] [AddCommMonoid R] [ContinuousAdd R]
    [ContinuousMul R] {A : X → Matrix m n R} {B : X → Matrix n p R} (hA : Continuous A)
    (hB : Continuous B) : Continuous fun x => A x * B x :=
  continuous_matrix fun _ _ =>
    continuous_finset_sum _ fun _ _ => (hA.matrix_elem _ _).mul (hB.matrix_elem _ _)


instance [Fintype n] [Mul R] [AddCommMonoid R] [ContinuousAdd R] [ContinuousMul R] :
    ContinuousMul (Matrix n n R) :=
  ⟨continuous_fst.matrix_mul continuous_snd⟩


instance [Fintype n] [NonUnitalNonAssocSemiring R] [TopologicalSemiring R] :
    TopologicalSemiring (Matrix n n R) where


instance Matrix.topologicalRing [Fintype n] [NonUnitalNonAssocRing R] [TopologicalRing R] :
    TopologicalRing (Matrix n n R) where


@[continuity]
theorem Continuous.matrix_vecMulVec [Mul R] [ContinuousMul R] {A : X → m → R} {B : X → n → R}
    (hA : Continuous A) (hB : Continuous B) : Continuous fun x => vecMulVec (A x) (B x) :=
  continuous_matrix fun _ _ => ((continuous_apply _).comp hA).mul ((continuous_apply _).comp hB)


@[continuity]
theorem Continuous.matrix_mulVec [NonUnitalNonAssocSemiring R] [ContinuousAdd R] [ContinuousMul R]
    [Fintype n] {A : X → Matrix m n R} {B : X → n → R} (hA : Continuous A) (hB : Continuous B) :
    Continuous fun x => A x *ᵥ B x :=
  continuous_pi fun i => ((continuous_apply i).comp hA).matrix_dotProduct hB


@[continuity]
theorem Continuous.matrix_vecMul [NonUnitalNonAssocSemiring R] [ContinuousAdd R] [ContinuousMul R]
    [Fintype m] {A : X → m → R} {B : X → Matrix m n R} (hA : Continuous A) (hB : Continuous B) :
    Continuous fun x => A x ᵥ* B x :=
  continuous_pi fun _i => hA.matrix_dotProduct <| continuous_pi fun _j => hB.matrix_elem _ _


@[continuity]
theorem Continuous.matrix_submatrix {A : X → Matrix l n R} (hA : Continuous A) (e₁ : m → l)
    (e₂ : p → n) : Continuous fun x => (A x).submatrix e₁ e₂ :=
  continuous_matrix fun _i _j => hA.matrix_elem _ _


@[continuity]
theorem Continuous.matrix_reindex {A : X → Matrix l n R} (hA : Continuous A) (e₁ : l ≃ m)
    (e₂ : n ≃ p) : Continuous fun x => reindex e₁ e₂ (A x) :=
  hA.matrix_submatrix _ _


@[continuity]
theorem Continuous.matrix_diag {A : X → Matrix n n R} (hA : Continuous A) :
    Continuous fun x => Matrix.diag (A x) :=
  continuous_pi fun _ => hA.matrix_elem _ _

-- note this doesn't elaborate well from the above

theorem continuous_matrix_diag : Continuous (Matrix.diag : Matrix n n R → n → R) :=
  show Continuous fun x : Matrix n n R => Matrix.diag x from continuous_id.matrix_diag


@[continuity]
theorem Continuous.matrix_trace [Fintype n] [AddCommMonoid R] [ContinuousAdd R]
    {A : X → Matrix n n R} (hA : Continuous A) : Continuous fun x => trace (A x) :=
  continuous_finset_sum _ fun _ _ => hA.matrix_elem _ _


@[continuity]
theorem Continuous.matrix_det [Fintype n] [DecidableEq n] [CommRing R] [TopologicalRing R]
    {A : X → Matrix n n R} (hA : Continuous A) : Continuous fun x => (A x).det := by
  /-
    X : Type u_1
    n : Type u_5
    R : Type u_8
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace R
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : TopologicalRing R
    A : X → Matrix n n R
    hA : Continuous A
    ⊢ Continuous fun x => (A x).det
  -/
  simp_rw [Matrix.det_apply]
  /-
    X : Type u_1
    n : Type u_5
    R : Type u_8
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace R
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : TopologicalRing R
    A : X → Matrix n n R
    hA : Continuous A
    ⊢ Continuous fun x => Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ) …
  -/
  refine continuous_finset_sum _ fun l _ => Continuous.const_smul ?_ _
  /-
    X : Type u_1
    n : Type u_5
    R : Type u_8
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace R
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : CommRing R
    inst✝ : TopologicalRing R
    A : X → Matrix n n R
    hA : Continuous A
    l : Equiv.Perm n
    x✝ : Membership.mem Finset.univ l
    ⊢ Continuous fun x => Finset.univ.prod fun i => A x (l i) i
  -/
  exact continuous_finset_prod _ fun l _ => hA.matrix_elem _ _
  /-
    🎉 no goals
  -/


@[continuity]
theorem Continuous.matrix_updateCol [DecidableEq n] (i : n) {A : X → Matrix m n R}
    {B : X → m → R} (hA : Continuous A) (hB : Continuous B) :
    Continuous fun x => (A x).updateCol i (B x) :=
  continuous_matrix fun _j k =>
    (continuous_apply k).comp <|
      ((continuous_apply _).comp hA).update i ((continuous_apply _).comp hB)


@[deprecated (since := "2024-12-11")]
alias Continuous.matrix_updateColumn := Continuous.matrix_updateCol


@[continuity]
theorem Continuous.matrix_updateRow [DecidableEq m] (i : m) {A : X → Matrix m n R} {B : X → n → R}
    (hA : Continuous A) (hB : Continuous B) : Continuous fun x => (A x).updateRow i (B x) :=
  hA.update i hB


@[continuity]
theorem Continuous.matrix_cramer [Fintype n] [DecidableEq n] [CommRing R] [TopologicalRing R]
    {A : X → Matrix n n R} {B : X → n → R} (hA : Continuous A) (hB : Continuous B) :
    Continuous fun x => cramer (A x) (B x) :=
  continuous_pi fun _ => (hA.matrix_updateCol _ hB).matrix_det


@[continuity]
theorem Continuous.matrix_adjugate [Fintype n] [DecidableEq n] [CommRing R] [TopologicalRing R]
    {A : X → Matrix n n R} (hA : Continuous A) : Continuous fun x => (A x).adjugate :=
  continuous_matrix fun _j k =>
    (hA.matrix_transpose.matrix_updateCol k continuous_const).matrix_det


/-- When `Ring.inverse` is continuous at the determinant (such as in a `NormedRing`, or a
topological field), so is `Matrix.inv`. -/
theorem continuousAt_matrix_inv [Fintype n] [DecidableEq n] [CommRing R] [TopologicalRing R]
    (A : Matrix n n R) (h : ContinuousAt Ring.inverse A.det) : ContinuousAt Inv.inv A :=
  (h.comp continuous_id.matrix_det.continuousAt).smul continuous_id.matrix_adjugate.continuousAt

-- lemmas about functions in `Data/Matrix/Block.lean`

@[continuity]
theorem Continuous.matrix_fromBlocks {A : X → Matrix n l R} {B : X → Matrix n m R}
    {C : X → Matrix p l R} {D : X → Matrix p m R} (hA : Continuous A) (hB : Continuous B)
    (hC : Continuous C) (hD : Continuous D) :
    Continuous fun x => Matrix.fromBlocks (A x) (B x) (C x) (D x) :=
  continuous_matrix <| by
    /-
      X : Type u_1
      l : Type u_3
      m : Type u_4
      n : Type u_5
      p : Type u_6
      R : Type u_8
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace R
      A : X → Matrix n l R
      B : X → Matrix n m R
      C : X → Matrix p l R
      D : X → Matrix p m R
      hA : Continuous A
      hB : Continuous B
      hC : Continuous C
      hD : Continuous D
      ⊢ ∀ (i : Sum n p) (j : Sum l m), Continuous fun a => Matrix.fromBlocks (A a) ( …
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    rintro (i | i) (j | j) <;> refine Continuous.matrix_elem ?_ i j <;> assumption
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[continuity]
theorem Continuous.matrix_blockDiagonal [Zero R] [DecidableEq p] {A : X → p → Matrix m n R}
    (hA : Continuous A) : Continuous fun x => blockDiagonal (A x) :=
  continuous_matrix fun ⟨i₁, i₂⟩ ⟨j₁, _j₂⟩ =>
    (((continuous_apply i₂).comp hA).matrix_elem i₁ j₁).if_const _ continuous_zero


@[continuity]
theorem Continuous.matrix_blockDiag {A : X → Matrix (m × p) (n × p) R} (hA : Continuous A) :
    Continuous fun x => blockDiag (A x) :=
  continuous_pi fun _i => continuous_matrix fun _j _k => hA.matrix_elem _ _


@[continuity]
theorem Continuous.matrix_blockDiagonal' [Zero R] [DecidableEq l]
    {A : X → ∀ i, Matrix (m' i) (n' i) R} (hA : Continuous A) :
    Continuous fun x => blockDiagonal' (A x) :=
  continuous_matrix fun ⟨i₁, i₂⟩ ⟨j₁, j₂⟩ => by
    /-
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace R
      inst✝¹ : Zero R
      inst✝ : DecidableEq l
      A : X → (i : l) → Matrix (m' i) (n' i) R
      hA : Continuous A
      x✝¹ : Sigma fun i => m' i
      x✝ : Sigma fun i => n' i
      i₁ : l
      i₂ : m' i₁
      j₁ : l
      j₂ : n' j₁
      ⊢ Continuous fun a => Matrix.blockDiagonal' (A a) ⟨i₁, i₂⟩ ⟨j₁, j₂⟩
    -/
    dsimp only [blockDiagonal'_apply']
    /-
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace R
      inst✝¹ : Zero R
      inst✝ : DecidableEq l
      A : X → (i : l) → Matrix (m' i) (n' i) R
      hA : Continuous A
      x✝¹ : Sigma fun i => m' i
      x✝ : Sigma fun i => n' i
      i₁ : l
      i₂ : m' i₁
      j₁ : l
      j₂ : n' j₁
      ⊢ Continuous fun a => dite (Eq i₁ j₁) (fun h => A a i₁ i₂ (cast ⋯ j₂)) fun h = …
    -/
    split_ifs with h
      /-
        case pos
        X : Type u_1
        l : Type u_3
        R : Type u_8
        m' : l → Type u_9
        n' : l → Type u_10
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace R
        inst✝¹ : Zero R
        inst✝ : DecidableEq l
        A : X → (i : l) → Matrix (m' i) (n' i) R
        hA : Continuous A
        x✝¹ : Sigma fun i => m' i
        x✝ : Sigma fun i => n' i
        i₁ : l
        i₂ : m' i₁
        j₁ : l
        j₂ : n' j₁
        h : Eq i₁ j₁
        ⊢ Continuous fun a => A a i₁ i₂ (cast ⋯ j₂)
      -/
    · subst h
      /-
        case pos
        X : Type u_1
        l : Type u_3
        R : Type u_8
        m' : l → Type u_9
        n' : l → Type u_10
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace R
        inst✝¹ : Zero R
        inst✝ : DecidableEq l
        A : X → (i : l) → Matrix (m' i) (n' i) R
        hA : Continuous A
        x✝¹ : Sigma fun i => m' i
        x✝ : Sigma fun i => n' i
        i₁ : l
        i₂ : m' i₁
        j₂ : n' i₁
        ⊢ Continuous fun a => A a i₁ i₂ (cast ⋯ j₂)
      -/
      exact ((continuous_apply i₁).comp hA).matrix_elem i₂ j₂
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        l : Type u_3
        R : Type u_8
        m' : l → Type u_9
        n' : l → Type u_10
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace R
        inst✝¹ : Zero R
        inst✝ : DecidableEq l
        A : X → (i : l) → Matrix (m' i) (n' i) R
        hA : Continuous A
        x✝¹ : Sigma fun i => m' i
        x✝ : Sigma fun i => n' i
        i₁ : l
        i₂ : m' i₁
        j₁ : l
        j₂ : n' j₁
        h : Not (Eq i₁ j₁)
        ⊢ Continuous fun a => 0
      -/
    · exact continuous_const
      /-
        🎉 no goals
      -/


@[continuity]
theorem Continuous.matrix_blockDiag' {A : X → Matrix (Σi, m' i) (Σi, n' i) R} (hA : Continuous A) :
    Continuous fun x => blockDiag' (A x) :=
  continuous_pi fun _i => continuous_matrix fun _j _k => hA.matrix_elem _ _


theorem HasSum.matrix_transpose {f : X → Matrix m n R} {a : Matrix m n R} (hf : HasSum f a) :
    HasSum (fun x => (f x)ᵀ) aᵀ :=
  (hf.map (Matrix.transposeAddEquiv m n R) continuous_id.matrix_transpose : _)


theorem Summable.matrix_transpose {f : X → Matrix m n R} (hf : Summable f) :
    Summable fun x => (f x)ᵀ :=
  hf.hasSum.matrix_transpose.summable


@[simp]
theorem summable_matrix_transpose {f : X → Matrix m n R} :
    (Summable fun x => (f x)ᵀ) ↔ Summable f :=
  Summable.map_iff_of_equiv (Matrix.transposeAddEquiv m n R)
    continuous_id.matrix_transpose continuous_id.matrix_transpose


theorem Matrix.transpose_tsum [T2Space R] {f : X → Matrix m n R} : (∑' x, f x)ᵀ = ∑' x, (f x)ᵀ := by
  /-
    X : Type u_1
    m : Type u_4
    n : Type u_5
    R : Type u_8
    inst✝² : AddCommMonoid R
    inst✝¹ : TopologicalSpace R
    inst✝ : T2Space R
    f : X → Matrix m n R
    ⊢ Eq (tsum fun x => f x).transpose (tsum fun x => (f x).transpose)
  -/
  by_cases hf : Summable f
    /-
      case pos
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝² : AddCommMonoid R
      inst✝¹ : TopologicalSpace R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Summable f
      ⊢ Eq (tsum fun x => f x).transpose (tsum fun x => (f x).transpose)
    -/
  · exact hf.hasSum.matrix_transpose.tsum_eq.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝² : AddCommMonoid R
      inst✝¹ : TopologicalSpace R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Not (Summable f)
      ⊢ Eq (tsum fun x => f x).transpose (tsum fun x => (f x).transpose)
    -/
  · have hft := summable_matrix_transpose.not.mpr hf
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝² : AddCommMonoid R
      inst✝¹ : TopologicalSpace R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Not (Summable f)
      hft : Not (Summable fun x => (f x).transpose)
      ⊢ Eq (tsum fun x => f x).transpose (tsum fun x => (f x).transpose)
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hft, transpose_zero]
    /-
      🎉 no goals
    -/


theorem HasSum.matrix_conjTranspose [StarAddMonoid R] [ContinuousStar R] {f : X → Matrix m n R}
    {a : Matrix m n R} (hf : HasSum f a) : HasSum (fun x => (f x)ᴴ) aᴴ :=
  (hf.map (Matrix.conjTransposeAddEquiv m n R) continuous_id.matrix_conjTranspose : _)


theorem Summable.matrix_conjTranspose [StarAddMonoid R] [ContinuousStar R] {f : X → Matrix m n R}
    (hf : Summable f) : Summable fun x => (f x)ᴴ :=
  hf.hasSum.matrix_conjTranspose.summable


@[simp]
theorem summable_matrix_conjTranspose [StarAddMonoid R] [ContinuousStar R] {f : X → Matrix m n R} :
    (Summable fun x => (f x)ᴴ) ↔ Summable f :=
  Summable.map_iff_of_equiv (Matrix.conjTransposeAddEquiv m n R)
    continuous_id.matrix_conjTranspose continuous_id.matrix_conjTranspose


theorem Matrix.conjTranspose_tsum [StarAddMonoid R] [ContinuousStar R] [T2Space R]
    {f : X → Matrix m n R} : (∑' x, f x)ᴴ = ∑' x, (f x)ᴴ := by
  /-
    X : Type u_1
    m : Type u_4
    n : Type u_5
    R : Type u_8
    inst✝⁴ : AddCommMonoid R
    inst✝³ : TopologicalSpace R
    inst✝² : StarAddMonoid R
    inst✝¹ : ContinuousStar R
    inst✝ : T2Space R
    f : X → Matrix m n R
    ⊢ Eq (tsum fun x => f x).conjTranspose (tsum fun x => (f x).conjTranspose)
  -/
  by_cases hf : Summable f
    /-
      case pos
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝⁴ : AddCommMonoid R
      inst✝³ : TopologicalSpace R
      inst✝² : StarAddMonoid R
      inst✝¹ : ContinuousStar R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Summable f
      ⊢ Eq (tsum fun x => f x).conjTranspose (tsum fun x => (f x).conjTranspose)
    -/
  · exact hf.hasSum.matrix_conjTranspose.tsum_eq.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝⁴ : AddCommMonoid R
      inst✝³ : TopologicalSpace R
      inst✝² : StarAddMonoid R
      inst✝¹ : ContinuousStar R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Not (Summable f)
      ⊢ Eq (tsum fun x => f x).conjTranspose (tsum fun x => (f x).conjTranspose)
    -/
  · have hft := summable_matrix_conjTranspose.not.mpr hf
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      R : Type u_8
      inst✝⁴ : AddCommMonoid R
      inst✝³ : TopologicalSpace R
      inst✝² : StarAddMonoid R
      inst✝¹ : ContinuousStar R
      inst✝ : T2Space R
      f : X → Matrix m n R
      hf : Not (Summable f)
      hft : Not (Summable fun x => (f x).conjTranspose)
      ⊢ Eq (tsum fun x => f x).conjTranspose (tsum fun x => (f x).conjTranspose)
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hft, conjTranspose_zero]
    /-
      🎉 no goals
    -/


theorem HasSum.matrix_diagonal [DecidableEq n] {f : X → n → R} {a : n → R} (hf : HasSum f a) :
    HasSum (fun x => diagonal (f x)) (diagonal a) :=
  hf.map (diagonalAddMonoidHom n R) continuous_id.matrix_diagonal


theorem Summable.matrix_diagonal [DecidableEq n] {f : X → n → R} (hf : Summable f) :
    Summable fun x => diagonal (f x) :=
  hf.hasSum.matrix_diagonal.summable


@[simp]
theorem summable_matrix_diagonal [DecidableEq n] {f : X → n → R} :
    (Summable fun x => diagonal (f x)) ↔ Summable f :=
  Summable.map_iff_of_leftInverse (Matrix.diagonalAddMonoidHom n R) (Matrix.diagAddMonoidHom n R)
    continuous_id.matrix_diagonal continuous_matrix_diag fun A => diag_diagonal A


theorem Matrix.diagonal_tsum [DecidableEq n] [T2Space R] {f : X → n → R} :
    diagonal (∑' x, f x) = ∑' x, diagonal (f x) := by
  /-
    X : Type u_1
    n : Type u_5
    R : Type u_8
    inst✝³ : AddCommMonoid R
    inst✝² : TopologicalSpace R
    inst✝¹ : DecidableEq n
    inst✝ : T2Space R
    f : X → n → R
    ⊢ Eq (Matrix.diagonal (tsum fun x => f x)) (tsum fun x => Matrix.diagonal (f x))
  -/
  by_cases hf : Summable f
    /-
      case pos
      X : Type u_1
      n : Type u_5
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq n
      inst✝ : T2Space R
      f : X → n → R
      hf : Summable f
      ⊢ Eq (Matrix.diagonal (tsum fun x => f x)) (tsum fun x => Matrix.diagonal (f x))
    -/
  · exact hf.hasSum.matrix_diagonal.tsum_eq.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      n : Type u_5
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq n
      inst✝ : T2Space R
      f : X → n → R
      hf : Not (Summable f)
      ⊢ Eq (Matrix.diagonal (tsum fun x => f x)) (tsum fun x => Matrix.diagonal (f x))
    -/
  · have hft := summable_matrix_diagonal.not.mpr hf
    /-
      case neg
      X : Type u_1
      n : Type u_5
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq n
      inst✝ : T2Space R
      f : X → n → R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.diagonal (f x))
      ⊢ Eq (Matrix.diagonal (tsum fun x => f x)) (tsum fun x => Matrix.diagonal (f x))
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hft]
    /-
      case neg
      X : Type u_1
      n : Type u_5
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq n
      inst✝ : T2Space R
      f : X → n → R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.diagonal (f x))
      ⊢ Eq (Matrix.diagonal 0) 0
    -/
    exact diagonal_zero
    /-
      🎉 no goals
    -/


theorem HasSum.matrix_diag {f : X → Matrix n n R} {a : Matrix n n R} (hf : HasSum f a) :
    HasSum (fun x => diag (f x)) (diag a) :=
  hf.map (diagAddMonoidHom n R) continuous_matrix_diag


theorem Summable.matrix_diag {f : X → Matrix n n R} (hf : Summable f) :
    Summable fun x => diag (f x) :=
  hf.hasSum.matrix_diag.summable


theorem HasSum.matrix_blockDiagonal [DecidableEq p] {f : X → p → Matrix m n R}
    {a : p → Matrix m n R} (hf : HasSum f a) :
    HasSum (fun x => blockDiagonal (f x)) (blockDiagonal a) :=
  hf.map (blockDiagonalAddMonoidHom m n p R) continuous_id.matrix_blockDiagonal


theorem Summable.matrix_blockDiagonal [DecidableEq p] {f : X → p → Matrix m n R} (hf : Summable f) :
    Summable fun x => blockDiagonal (f x) :=
  hf.hasSum.matrix_blockDiagonal.summable


theorem summable_matrix_blockDiagonal [DecidableEq p] {f : X → p → Matrix m n R} :
    (Summable fun x => blockDiagonal (f x)) ↔ Summable f :=
  Summable.map_iff_of_leftInverse (blockDiagonalAddMonoidHom m n p R)
    (blockDiagAddMonoidHom m n p R) continuous_id.matrix_blockDiagonal
    continuous_id.matrix_blockDiag fun A => blockDiag_blockDiagonal A


theorem Matrix.blockDiagonal_tsum [DecidableEq p] [T2Space R] {f : X → p → Matrix m n R} :
    blockDiagonal (∑' x, f x) = ∑' x, blockDiagonal (f x) := by
  /-
    X : Type u_1
    m : Type u_4
    n : Type u_5
    p : Type u_6
    R : Type u_8
    inst✝³ : AddCommMonoid R
    inst✝² : TopologicalSpace R
    inst✝¹ : DecidableEq p
    inst✝ : T2Space R
    f : X → p → Matrix m n R
    ⊢ Eq (Matrix.blockDiagonal (tsum fun x => f x)) (tsum fun x => Matrix.blockDia …
  -/
  by_cases hf : Summable f
    /-
      case pos
      X : Type u_1
      m : Type u_4
      n : Type u_5
      p : Type u_6
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq p
      inst✝ : T2Space R
      f : X → p → Matrix m n R
      hf : Summable f
      ⊢ Eq (Matrix.blockDiagonal (tsum fun x => f x)) (tsum fun x => Matrix.blockDia …
    -/
  · exact hf.hasSum.matrix_blockDiagonal.tsum_eq.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      p : Type u_6
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq p
      inst✝ : T2Space R
      f : X → p → Matrix m n R
      hf : Not (Summable f)
      ⊢ Eq (Matrix.blockDiagonal (tsum fun x => f x)) (tsum fun x => Matrix.blockDia …
    -/
  · have hft := summable_matrix_blockDiagonal.not.mpr hf
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      p : Type u_6
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq p
      inst✝ : T2Space R
      f : X → p → Matrix m n R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.blockDiagonal (f x))
      ⊢ Eq (Matrix.blockDiagonal (tsum fun x => f x)) (tsum fun x => Matrix.blockDia …
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hft]
    /-
      case neg
      X : Type u_1
      m : Type u_4
      n : Type u_5
      p : Type u_6
      R : Type u_8
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq p
      inst✝ : T2Space R
      f : X → p → Matrix m n R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.blockDiagonal (f x))
      ⊢ Eq (Matrix.blockDiagonal 0) 0
    -/
    exact blockDiagonal_zero
    /-
      🎉 no goals
    -/


theorem HasSum.matrix_blockDiag {f : X → Matrix (m × p) (n × p) R} {a : Matrix (m × p) (n × p) R}
    (hf : HasSum f a) : HasSum (fun x => blockDiag (f x)) (blockDiag a) :=
  (hf.map (blockDiagAddMonoidHom m n p R) <| Continuous.matrix_blockDiag continuous_id : _)


theorem Summable.matrix_blockDiag {f : X → Matrix (m × p) (n × p) R} (hf : Summable f) :
    Summable fun x => blockDiag (f x) :=
  hf.hasSum.matrix_blockDiag.summable


theorem HasSum.matrix_blockDiagonal' [DecidableEq l] {f : X → ∀ i, Matrix (m' i) (n' i) R}
    {a : ∀ i, Matrix (m' i) (n' i) R} (hf : HasSum f a) :
    HasSum (fun x => blockDiagonal' (f x)) (blockDiagonal' a) :=
  hf.map (blockDiagonal'AddMonoidHom m' n' R) continuous_id.matrix_blockDiagonal'


theorem Summable.matrix_blockDiagonal' [DecidableEq l] {f : X → ∀ i, Matrix (m' i) (n' i) R}
    (hf : Summable f) : Summable fun x => blockDiagonal' (f x) :=
  hf.hasSum.matrix_blockDiagonal'.summable


theorem summable_matrix_blockDiagonal' [DecidableEq l] {f : X → ∀ i, Matrix (m' i) (n' i) R} :
    (Summable fun x => blockDiagonal' (f x)) ↔ Summable f :=
  Summable.map_iff_of_leftInverse (blockDiagonal'AddMonoidHom m' n' R)
    (blockDiag'AddMonoidHom m' n' R) continuous_id.matrix_blockDiagonal'
    continuous_id.matrix_blockDiag' fun A => blockDiag'_blockDiagonal' A


theorem Matrix.blockDiagonal'_tsum [DecidableEq l] [T2Space R]
    {f : X → ∀ i, Matrix (m' i) (n' i) R} :
    blockDiagonal' (∑' x, f x) = ∑' x, blockDiagonal' (f x) := by
  /-
    X : Type u_1
    l : Type u_3
    R : Type u_8
    m' : l → Type u_9
    n' : l → Type u_10
    inst✝³ : AddCommMonoid R
    inst✝² : TopologicalSpace R
    inst✝¹ : DecidableEq l
    inst✝ : T2Space R
    f : X → (i : l) → Matrix (m' i) (n' i) R
    ⊢ Eq (Matrix.blockDiagonal' (tsum fun x => f x)) (tsum fun x => Matrix.blockDi …
  -/
  by_cases hf : Summable f
    /-
      case pos
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq l
      inst✝ : T2Space R
      f : X → (i : l) → Matrix (m' i) (n' i) R
      hf : Summable f
      ⊢ Eq (Matrix.blockDiagonal' (tsum fun x => f x)) (tsum fun x => Matrix.blockDi …
    -/
  · exact hf.hasSum.matrix_blockDiagonal'.tsum_eq.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq l
      inst✝ : T2Space R
      f : X → (i : l) → Matrix (m' i) (n' i) R
      hf : Not (Summable f)
      ⊢ Eq (Matrix.blockDiagonal' (tsum fun x => f x)) (tsum fun x => Matrix.blockDi …
    -/
  · have hft := summable_matrix_blockDiagonal'.not.mpr hf
    /-
      case neg
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq l
      inst✝ : T2Space R
      f : X → (i : l) → Matrix (m' i) (n' i) R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.blockDiagonal' (f x))
      ⊢ Eq (Matrix.blockDiagonal' (tsum fun x => f x)) (tsum fun x => Matrix.blockDi …
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hft]
    /-
      case neg
      X : Type u_1
      l : Type u_3
      R : Type u_8
      m' : l → Type u_9
      n' : l → Type u_10
      inst✝³ : AddCommMonoid R
      inst✝² : TopologicalSpace R
      inst✝¹ : DecidableEq l
      inst✝ : T2Space R
      f : X → (i : l) → Matrix (m' i) (n' i) R
      hf : Not (Summable f)
      hft : Not (Summable fun x => Matrix.blockDiagonal' (f x))
      ⊢ Eq (Matrix.blockDiagonal' 0) 0
    -/
    exact blockDiagonal'_zero
    /-
      🎉 no goals
    -/


theorem HasSum.matrix_blockDiag' {f : X → Matrix (Σi, m' i) (Σi, n' i) R}
    {a : Matrix (Σi, m' i) (Σi, n' i) R} (hf : HasSum f a) :
    HasSum (fun x => blockDiag' (f x)) (blockDiag' a) :=
  hf.map (blockDiag'AddMonoidHom m' n' R) continuous_id.matrix_blockDiag'


theorem Summable.matrix_blockDiag' {f : X → Matrix (Σi, m' i) (Σi, n' i) R} (hf : Summable f) :
    Summable fun x => blockDiag' (f x) :=
  hf.hasSum.matrix_blockDiag'.summable


