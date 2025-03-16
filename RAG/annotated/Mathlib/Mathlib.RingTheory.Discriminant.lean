/-- Given an `A`-algebra `B` and `b`, an `ι`-indexed family of elements of `B`, we define
`discr A ι b` as the determinant of `traceMatrix A ι b`. -/
-- Porting note: using `[DecidableEq ι]` instead of `by classical...` did not work in
-- mathlib3.
noncomputable def discr (A : Type u) {B : Type v} [CommRing A] [CommRing B] [Algebra A B]
    [Fintype ι] (b : ι → B) := (traceMatrix A b).det


theorem discr_def [Fintype ι] (b : ι → B) : discr A b = (traceMatrix A b).det := rfl


variable {A C} in
/-- Mapping a family of vectors along an `AlgEquiv` preserves the discriminant. -/
theorem discr_eq_discr_of_algEquiv [Fintype ι] (b : ι → B) (f : B ≃ₐ[A] C) :
    Algebra.discr A b = Algebra.discr A (f ∘ b) := by
  /-
    A : Type u
    B : Type v
    C : Type z
    ι : Type w
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : CommRing C
    inst✝¹ : Algebra A C
    inst✝ : Fintype ι
    b : ι → B
    f : AlgEquiv A B C
    ⊢ Eq (Algebra.discr A b) (Algebra.discr A (Function.comp (⇑f) b))
  -/
  rw [discr_def]; congr; ext
  /-
    case e_M.a
    A : Type u
    B : Type v
    C : Type z
    ι : Type w
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : CommRing C
    inst✝¹ : Algebra A C
    inst✝ : Fintype ι
    b : ι → B
    f : AlgEquiv A B C
    i✝ j✝ : ι
    ⊢ Eq (Algebra.traceMatrix A b i✝ j✝) (Algebra.traceMatrix A (Function.comp (⇑f …
  -/
  simp_rw [traceMatrix_apply, traceForm_apply, Function.comp, ← map_mul f, trace_eq_of_algEquiv]
  /-
    🎉 no goals
  -/


@[simp]
theorem discr_reindex (b : Basis ι A B) (f : ι ≃ ι') : discr A (b ∘ ⇑f.symm) = discr A b := by
  /-
    A : Type u
    B : Type v
    ι : Type w
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    ι' : Type u_1
    inst✝² : Fintype ι'
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι'
    b : Basis ι A B
    f : Equiv ι ι'
    ⊢ Eq (Algebra.discr A (Function.comp ⇑b ⇑f.symm)) (Algebra.discr A ⇑b)
  -/
  classical rw [← Basis.coe_reindex, discr_def, traceMatrix_reindex, det_reindex_self, ← discr_def]
  /-
    🎉 no goals
  -/


/-- If `b` is not linear independent, then `Algebra.discr A b = 0`. -/
theorem discr_zero_of_not_linearIndependent [IsDomain A] {b : ι → B}
    (hli : ¬LinearIndependent A b) : discr A b = 0 := by
  classical
  obtain ⟨g, hg, i, hi⟩ := Fintype.not_linearIndependent_iff.1 hli
  have : (traceMatrix A b) *ᵥ g = 0 := by
    ext i
    have : ∀ j, (trace A B) (b i * b j) * g j = (trace A B) (g j • b j * b i) := by
      intro j
      simp [mul_comm]
    simp only [mulVec, dotProduct, traceMatrix_apply, Pi.zero_apply, traceForm_apply, fun j =>
      this j, ← map_sum, ← sum_mul, hg, zero_mul, LinearMap.map_zero]
  by_contra h
  rw [discr_def] at h
  simp [Matrix.eq_zero_of_mulVec_eq_zero h this] at hi


/-- Relation between `Algebra.discr A ι b` and
`Algebra.discr A (b ᵥ* P.map (algebraMap A B))`. -/
theorem discr_of_matrix_vecMul (b : ι → B) (P : Matrix ι ι A) :
    discr A (b ᵥ* P.map (algebraMap A B)) = P.det ^ 2 * discr A b := by
  rw [discr_def, traceMatrix_of_matrix_vecMul, det_mul, det_mul, det_transpose, mul_comm, ←
    mul_assoc, discr_def, pow_two]


/-- Relation between `Algebra.discr A ι b` and
`Algebra.discr A ((P.map (algebraMap A B)) *ᵥ b)`. -/
theorem discr_of_matrix_mulVec (b : ι → B) (P : Matrix ι ι A) :
    discr A (P.map (algebraMap A B) *ᵥ b) = P.det ^ 2 * discr A b := by
  rw [discr_def, traceMatrix_of_matrix_mulVec, det_mul, det_mul, det_transpose, mul_comm, ←
    mul_assoc, discr_def, pow_two]


/-- If `b` is a basis of a finite separable field extension `L/K`, then `Algebra.discr K b ≠ 0`. -/
theorem discr_not_zero_of_basis [Algebra.IsSeparable K L] (b : Basis ι K L) :
    discr K b ≠ 0 := by
  /-
    ι : Type w
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Fintype ι
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    inst✝ : Algebra.IsSeparable K L
    b : Basis ι K L
    ⊢ Ne (Algebra.discr K ⇑b) 0
  -/
  rw [discr_def, traceMatrix_of_basis, ← LinearMap.BilinForm.nondegenerate_iff_det_ne_zero]
  /-
    ι : Type w
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Fintype ι
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    inst✝ : Algebra.IsSeparable K L
    b : Basis ι K L
    ⊢ (Algebra.traceForm K L).Nondegenerate
  -/
  exact traceForm_nondegenerate _ _
  /-
    🎉 no goals
  -/


/-- If `b` is a basis of a finite separable field extension `L/K`,
  then `Algebra.discr K b` is a unit. -/
theorem discr_isUnit_of_basis [Algebra.IsSeparable K L] (b : Basis ι K L) : IsUnit (discr K b) :=
  IsUnit.mk0 _ (discr_not_zero_of_basis _ _)


/-- If `L/K` is a field extension and `b : ι → L`, then `discr K b` is the square of the
determinant of the matrix whose `(i, j)` coefficient is `σⱼ (b i)`, where `σⱼ : L →ₐ[K] E` is the
embedding in an algebraically closed field `E` corresponding to `j : ι` via a bijection
`e : ι ≃ (L →ₐ[K] E)`. -/
theorem discr_eq_det_embeddingsMatrixReindex_pow_two
    [Algebra.IsSeparable K L] (e : ι ≃ (L →ₐ[K] E)) :
    algebraMap K E (discr K b) = (embeddingsMatrixReindex K E b e).det ^ 2 := by
  rw [discr_def, RingHom.map_det, RingHom.mapMatrix_apply,
    traceMatrix_eq_embeddingsMatrixReindex_mul_trans, det_mul, det_transpose, pow_two]


/-- The discriminant of a power basis. -/
theorem discr_powerBasis_eq_prod (e : Fin pb.dim ≃ (L →ₐ[K] E)) [Algebra.IsSeparable K L] :
    algebraMap K E (discr K pb.basis) =
      ∏ i : Fin pb.dim, ∏ j ∈ Ioi i, (e j pb.gen - e i pb.gen) ^ 2 := by
  rw [discr_eq_det_embeddingsMatrixReindex_pow_two K E pb.basis e,
    embeddingsMatrixReindex_eq_vandermonde, det_transpose, det_vandermonde, ← prod_pow]
  /-
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    inst✝ : Algebra.IsSeparable K L
    ⊢ Eq (Finset.univ.prod fun x => HPow.hPow ((Finset.Ioi x).prod fun j => HSub.h …
  -/
  congr; ext i
  /-
    case e_f.h
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    inst✝ : Algebra.IsSeparable K L
    i : Fin pb.dim
    ⊢ Eq (HPow.hPow ((Finset.Ioi i).prod fun j => HSub.hSub ((e j) pb.gen) ((e i)  …
  -/
  rw [← prod_pow]
  /-
    🎉 no goals
  -/


/-- A variation of `Algebra.discr_powerBasis_eq_prod`. -/
theorem discr_powerBasis_eq_prod' [Algebra.IsSeparable K L] (e : Fin pb.dim ≃ (L →ₐ[K] E)) :
    algebraMap K E (discr K pb.basis) =
      ∏ i : Fin pb.dim, ∏ j ∈ Ioi i, -((e j pb.gen - e i pb.gen) * (e i pb.gen - e j pb.gen)) := by
  /-
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq ((algebraMap K E) (Algebra.discr K ⇑pb.basis)) (Finset.univ.prod fun i => …
  -/
  rw [discr_powerBasis_eq_prod _ _ _ e]
  /-
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (Finset.univ.prod fun i => (Finset.Ioi i).prod fun j => HPow.hPow (HSub.h …
  -/
  congr; ext i; congr; ext j
  /-
    case e_f.h.e_f.h
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    i j : Fin pb.dim
    ⊢ Eq (HPow.hPow (HSub.hSub ((e j) pb.gen) ((e i) pb.gen)) 2) (Neg.neg (HMul.hM …
  -/
  ring
  /-
    🎉 no goals
  -/


local notation "n" => finrank K L


/-- A variation of `Algebra.discr_powerBasis_eq_prod`. -/
theorem discr_powerBasis_eq_prod'' [Algebra.IsSeparable K L] (e : Fin pb.dim ≃ (L →ₐ[K] E)) :
    algebraMap K E (discr K pb.basis) =
      (-1) ^ (n * (n - 1) / 2) *
        ∏ i : Fin pb.dim, ∏ j ∈ Ioi i, (e j pb.gen - e i pb.gen) * (e i pb.gen - e j pb.gen) := by
  /-
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq ((algebraMap K E) (Algebra.discr K ⇑pb.basis)) (HMul.hMul (HPow.hPow (-1) …
  -/
  rw [discr_powerBasis_eq_prod' _ _ _ e]
  simp_rw [fun i j => neg_eq_neg_one_mul ((e j pb.gen - e i pb.gen) * (e i pb.gen - e j pb.gen)),
    prod_mul_distrib]
  /-
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun x => (Finset.Ioi x).prod fun x => -1) (H …
  -/
  congr
  /-
    case e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (Finset.univ.prod fun x => (Finset.Ioi x).prod fun x => -1) (HPow.hPow (- …
  -/
  simp only [prod_pow_eq_pow_sum, prod_const]
  /-
    case e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (HPow.hPow (-1) (Finset.univ.sum fun i => (Finset.Ioi i).card)) (HPow.hPo …
  -/
  congr
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (Finset.univ.sum fun i => (Finset.Ioi i).card) (HDiv.hDiv (HMul.hMul (Mod …
  -/
  rw [← @Nat.cast_inj ℚ, Nat.cast_sum]
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    ⊢ Eq (Finset.univ.sum fun x => ↑(Finset.Ioi x).card) ↑(HDiv.hDiv (HMul.hMul (M …
  -/
  have : ∀ x : Fin pb.dim, ↑x + 1 ≤ pb.dim := by simp [Nat.succ_le_iff, Fin.is_lt]
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    ⊢ Eq (Finset.univ.sum fun x => ↑(Finset.Ioi x).card) ↑(HDiv.hDiv (HMul.hMul (M …
  -/
  simp_rw [Fin.card_Ioi, Nat.sub_sub, add_comm 1]
  simp only [Nat.cast_sub, this, Finset.card_fin, nsmul_eq_mul, sum_const, sum_sub_distrib,
    Nat.cast_add, Nat.cast_one, sum_add_distrib, mul_one]
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    ⊢ Eq (HSub.hSub (HMul.hMul ↑pb.dim ↑pb.dim) (HAdd.hAdd (Finset.univ.sum fun x  …
  -/
  rw [← Nat.cast_sum, ← @Finset.sum_range ℕ _ pb.dim fun i => i, sum_range_id]
  have hn : n = pb.dim := by
    rw [← AlgHom.card K L E, ← Fintype.card_fin pb.dim]
    -- FIXME: Without the `Fintype` namespace, why does it complain about `Finset.card_congr` being
    -- deprecated?
    exact Fintype.card_congr e.symm
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    hn : Eq (Module.finrank K L) pb.dim
    ⊢ Eq (HSub.hSub (HMul.hMul ↑pb.dim ↑pb.dim) (HAdd.hAdd ↑(HDiv.hDiv (HMul.hMul  …
  -/
  have h₂ : 2 ∣ pb.dim * (pb.dim - 1) := pb.dim.even_mul_pred_self.two_dvd
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    hn : Eq (Module.finrank K L) pb.dim
    h₂ : Dvd.dvd 2 (HMul.hMul pb.dim (HSub.hSub pb.dim 1))
    ⊢ Eq (HSub.hSub (HMul.hMul ↑pb.dim ↑pb.dim) (HAdd.hAdd ↑(HDiv.hDiv (HMul.hMul  …
  -/
  have hne : ((2 : ℕ) : ℚ) ≠ 0 := by simp
  have hle : 1 ≤ pb.dim := by
    rw [← hn, Nat.one_le_iff_ne_zero, ← zero_lt_iff, Module.finrank_pos_iff]
    infer_instance
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    hn : Eq (Module.finrank K L) pb.dim
    h₂ : Dvd.dvd 2 (HMul.hMul pb.dim (HSub.hSub pb.dim 1))
    hne : Ne (↑2) 0
    hle : LE.le 1 pb.dim
    ⊢ Eq (HSub.hSub (HMul.hMul ↑pb.dim ↑pb.dim) (HAdd.hAdd ↑(HDiv.hDiv (HMul.hMul  …
  -/
  rw [hn, Nat.cast_div h₂ hne, Nat.cast_mul, Nat.cast_sub hle]
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    hn : Eq (Module.finrank K L) pb.dim
    h₂ : Dvd.dvd 2 (HMul.hMul pb.dim (HSub.hSub pb.dim 1))
    hne : Ne (↑2) 0
    hle : LE.le 1 pb.dim
    ⊢ Eq (HSub.hSub (HMul.hMul ↑pb.dim ↑pb.dim) (HAdd.hAdd (HDiv.hDiv (HMul.hMul ( …
  -/
  field_simp
  /-
    case e_a.e_a
    K : Type u
    L : Type v
    E : Type z
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Field E
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : IsAlgClosed E
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    this : ∀ (x : Fin pb.dim), LE.le (HAdd.hAdd (↑x) 1) pb.dim
    hn : Eq (Module.finrank K L) pb.dim
    h₂ : Dvd.dvd 2 (HMul.hMul pb.dim (HSub.hSub pb.dim 1))
    hne : Ne (↑2) 0
    hle : LE.le 1 pb.dim
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul ↑pb.dim ↑pb.dim) 2) (HAdd.hAdd (HMul.hMu …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Formula for the discriminant of a power basis using the norm of the field extension. -/
-- Porting note: `(minpoly K pb.gen).derivative` does not work anymore.
theorem discr_powerBasis_eq_norm [Algebra.IsSeparable K L] :
    discr K pb.basis =
      (-1) ^ (n * (n - 1) / 2) *
      norm K (aeval pb.gen (derivative (R := K) (minpoly K pb.gen))) := by
  /-
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Eq (Algebra.discr K ⇑pb.basis) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.h …
  -/
  let E := AlgebraicClosure L
  /-
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    E : Type v := AlgebraicClosure L
    ⊢ Eq (Algebra.discr K ⇑pb.basis) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.h …
  -/
  letI := fun a b : E => Classical.propDecidable (Eq a b)
  have e : Fin pb.dim ≃ (L →ₐ[K] E) := by
    refine equivOfCardEq ?_
    rw [Fintype.card_fin, AlgHom.card]
    exact (PowerBasis.finrank pb).symm
  have hnodup : ((minpoly K pb.gen).aroots E).Nodup :=
    nodup_roots (Separable.map (Algebra.IsSeparable.isSeparable K pb.gen))
  have hroots : ∀ σ : L →ₐ[K] E, σ pb.gen ∈ (minpoly K pb.gen).aroots E := by
    intro σ
    rw [mem_roots, IsRoot.def, eval_map, ← aeval_def, aeval_algHom_apply]
    repeat' simp [minpoly.ne_zero (Algebra.IsSeparable.isIntegral K pb.gen)]
  /-
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    E : Type v := AlgebraicClosure L
    this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    hnodup : ((minpoly K pb.gen).aroots E).Nodup
    hroots : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) (σ …
    ⊢ Eq (Algebra.discr K ⇑pb.basis) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.h …
  -/
  apply (algebraMap K E).injective
  rw [RingHom.map_mul, RingHom.map_pow, RingHom.map_neg, RingHom.map_one,
    discr_powerBasis_eq_prod'' _ _ _ e]
  /-
    case a
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    E : Type v := AlgebraicClosure L
    this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    hnodup : ((minpoly K pb.gen).aroots E).Nodup
    hroots : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) (σ …
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (Module.finrank K L) (HS …
  -/
  congr
  /-
    case a.e_a
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    E : Type v := AlgebraicClosure L
    this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    hnodup : ((minpoly K pb.gen).aroots E).Nodup
    hroots : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) (σ …
    ⊢ Eq (Finset.univ.prod fun i => (Finset.Ioi i).prod fun j => HMul.hMul (HSub.h …
  -/
  rw [norm_eq_prod_embeddings, prod_prod_Ioi_mul_eq_prod_prod_off_diag]
  conv_rhs =>
    congr
    rfl
    ext σ
    rw [← aeval_algHom_apply,
      aeval_root_derivative_of_splits (minpoly.monic (Algebra.IsSeparable.isIntegral K pb.gen))
        (IsAlgClosed.splits_codomain _) (hroots σ),
      ← Finset.prod_mk _ (hnodup.erase _)]
  /-
    case a.e_a
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Module.Finite K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    E : Type v := AlgebraicClosure L
    this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
    e : Equiv (Fin pb.dim) (AlgHom K L E)
    hnodup : ((minpoly K pb.gen).aroots E).Nodup
    hroots : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) (σ …
    ⊢ Eq (Finset.univ.prod fun i => (HasCompl.compl (Singleton.singleton i)).prod  …
  -/
  rw [prod_sigma', prod_sigma']
  refine prod_bij' (fun i _ ↦ ⟨e i.2, e i.1 pb.gen⟩)
    (fun σ hσ ↦ ⟨e.symm (PowerBasis.lift pb σ.2 ?_), e.symm σ.1⟩) ?_ ?_ ?_ ?_ (fun i _ ↦ by simp)
  -- Porting note: `@mem_compl` was not necessary.
    <;> simp only [mem_sigma, mem_univ, Finset.mem_mk, hnodup.mem_erase_iff, IsRoot.def,
      mem_roots', minpoly.ne_zero (Algebra.IsSeparable.isIntegral K pb.gen), not_false_eq_true,
      mem_singleton, true_and, @mem_compl _ _ _ (_), Sigma.forall, Equiv.apply_symm_apply,
      PowerBasis.lift_gen, and_imp, implies_true, forall_const, Equiv.symm_apply_apply,
      Sigma.ext_iff, Equiv.symm_apply_eq, heq_eq_eq, and_true] at *
    /-
      case a.e_a.refine_1
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      σ : Sigma fun i => E
      hroots : ∀ (σ : AlgHom K L E), And (Ne (Polynomial.map (algebraMap K E) (minpo …
      hσ : And (Ne σ.snd (σ.fst pb.gen)) (And (Ne (Polynomial.map (algebraMap K E) ( …
      ⊢ Eq ((Polynomial.aeval σ.snd) (minpoly K pb.gen)) 0
    -/
  · simpa only [aeval_def, eval₂_eq_eval_map] using hσ.2.2
    /-
      🎉 no goals
    -/
    /-
      case a.e_a.refine_2
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      hroots : ∀ (σ : AlgHom K L E), And (Ne (Polynomial.map (algebraMap K E) (minpo …
      ⊢ ∀ (a b : Fin pb.dim), Not (Eq b a) → And (Ne ((e a) pb.gen) ((e b) pb.gen))  …
    -/
  · exact fun a b hba ↦ ⟨fun h ↦ hba <| e.injective <| pb.algHom_ext h.symm, hroots _⟩
    /-
      🎉 no goals
    -/
    /-
      case a.e_a.refine_3
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      hroots✝ : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) ( …
      hroots : ∀ (σ : AlgHom K L E), And (Ne (Polynomial.map (algebraMap K E) (minpo …
      ⊢ ∀ (a : AlgHom K L E) (b : E) (ha : And (Ne b (a pb.gen)) (And (Ne (Polynomia …
    -/
  · rintro a b hba ha
    /-
      case a.e_a.refine_3
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      hroots✝ : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) ( …
      hroots : ∀ (σ : AlgHom K L E), And (Ne (Polynomial.map (algebraMap K E) (minpo …
      a : AlgHom K L E
      b : E
      hba : And (Ne b (a pb.gen)) (And (Ne (Polynomial.map (algebraMap K E) (minpoly …
      ha : Eq a (pb.lift b ⋯)
      ⊢ False
    -/
    rw [ha, PowerBasis.lift_gen] at hba
    /-
      case a.e_a.refine_3
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      hroots✝ : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) ( …
      hroots : ∀ (σ : AlgHom K L E), And (Ne (Polynomial.map (algebraMap K E) (minpo …
      a : AlgHom K L E
      b : E
      hba✝ : And (Ne b (a pb.gen)) (And (Ne (Polynomial.map (algebraMap K E) (minpol …
      hba : And (Ne b b) (And (Ne (Polynomial.map (algebraMap K E) (minpoly K pb.gen …
      ha : Eq a (pb.lift b ⋯)
      ⊢ False
    -/
    exact hba.1 rfl
    /-
      🎉 no goals
    -/
    /-
      case a.e_a.refine_4
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : Module.Finite K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      E : Type v := AlgebraicClosure L
      this : (a b : E) → Decidable (Eq a b) := fun a b => Classical.propDecidable (E …
      e : Equiv (Fin pb.dim) (AlgHom K L E)
      hnodup : ((minpoly K pb.gen).aroots E).Nodup
      hroots : ∀ (σ : AlgHom K L E), Membership.mem ((minpoly K pb.gen).aroots E) (σ …
      ⊢ ∀ (a b : Fin pb.dim) (ha : Not (Eq b a)), Eq (pb.lift ((e a) pb.gen) ⋯) (e a)
    -/
  · exact fun a b _ ↦ pb.algHom_ext <| pb.lift_gen _ _
    /-
      🎉 no goals
    -/


/-- If `K` and `L` are fields and `IsScalarTower R K L`, and `b : ι → L` satisfies
` ∀ i, IsIntegral R (b i)`, then `IsIntegral R (discr K b)`. -/
theorem discr_isIntegral {b : ι → L} (h : ∀ i, IsIntegral R (b i)) : IsIntegral R (discr K b) := by
  classical
  rw [discr_def]
  exact IsIntegral.det fun i j ↦ isIntegral_trace ((h i).mul (h j))


/-- Let `K` be the fraction field of an integrally closed domain `R` and let `L` be a finite
separable extension of `K`. Let `B : PowerBasis K L` be such that `IsIntegral R B.gen`.
Then for all, `z : L` that are integral over `R`, we have
`(discr K B.basis) • z ∈ adjoin R ({B.gen} : Set L)`. -/
theorem discr_mul_isIntegral_mem_adjoin [Algebra.IsSeparable K L] [IsIntegrallyClosed R]
    [IsFractionRing R K] {B : PowerBasis K L} (hint : IsIntegral R B.gen) {z : L}
    (hz : IsIntegral R z) : discr K B.basis • z ∈ adjoin R ({B.gen} : Set L) := by
  have hinv : IsUnit (traceMatrix K B.basis).det := by
    simpa [← discr_def] using discr_isUnit_of_basis _ B.basis
  have H :
    (traceMatrix K B.basis).det • (traceMatrix K B.basis) *ᵥ (B.basis.equivFun z) =
      (traceMatrix K B.basis).det • fun i => trace K L (z * B.basis i) := by
    congr; exact traceMatrix_of_basis_mulVec _ _
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton B.gen)) (HSMul.hSMul ( …
  -/
  have cramer := mulVec_cramer (traceMatrix K B.basis) fun i => trace K L (z * B.basis i)
  suffices ∀ i, ((traceMatrix K B.basis).det • B.basis.equivFun z) i ∈ (⊥ : Subalgebra R K) by
    rw [← B.basis.sum_repr z, Finset.smul_sum]
    refine Subalgebra.sum_mem _ fun i _ => ?_
    replace this := this i
    rw [← discr_def, Pi.smul_apply, mem_bot] at this
    obtain ⟨r, hr⟩ := this
    rw [Basis.equivFun_apply] at hr
    rw [← smul_assoc, ← hr, algebraMap_smul]
    refine Subalgebra.smul_mem _ ?_ _
    rw [B.basis_eq_pow i]
    exact Subalgebra.pow_mem _ (subset_adjoin (Set.mem_singleton _)) _
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).mulVec ((Algebra.traceMatrix K ⇑ …
    ⊢ ∀ (i : Fin B.dim), Membership.mem Bot.bot (HSMul.hSMul (Algebra.traceMatrix  …
  -/
  intro i
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).mulVec ((Algebra.traceMatrix K ⇑ …
    i : Fin B.dim
    ⊢ Membership.mem Bot.bot (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det (B. …
  -/
  rw [← H, ← mulVec_smul] at cramer
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).mulVec ((Algebra.traceMatrix K ⇑ …
    i : Fin B.dim
    ⊢ Membership.mem Bot.bot (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det (B. …
  -/
  replace cramer := congr_arg (mulVec (traceMatrix K B.basis)⁻¹) cramer
  rw [mulVec_mulVec, nonsing_inv_mul _ hinv, mulVec_mulVec, nonsing_inv_mul _ hinv, one_mulVec,
    one_mulVec] at cramer
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    i : Fin B.dim
    cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).cramer fun i => (Algebra.trace K …
    ⊢ Membership.mem Bot.bot (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det (B. …
  -/
  rw [← congr_fun cramer i, cramer_apply, det_apply]
  refine
    Subalgebra.sum_mem _ fun σ _ => Subalgebra.zsmul_mem _ (Subalgebra.prod_mem _ fun j _ => ?_) _
  /-
    K : Type u
    L : Type v
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module.Finite K L
    R : Type z
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R K L
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsFractionRing R K
    B : PowerBasis K L
    hint : IsIntegral R B.gen
    z : L
    hz : IsIntegral R z
    hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
    H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
    i : Fin B.dim
    cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).cramer fun i => (Algebra.trace K …
    σ : Equiv.Perm (Fin B.dim)
    x✝¹ : Membership.mem Finset.univ σ
    j : Fin B.dim
    x✝ : Membership.mem Finset.univ j
    ⊢ Membership.mem Bot.bot ((Algebra.traceMatrix K ⇑B.basis).updateCol i (fun i  …
  -/
  by_cases hji : j = i
    /-
      case pos
      K : Type u
      L : Type v
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra K L
      inst✝⁷ : Module.Finite K L
      R : Type z
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R K
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R K L
      inst✝² : Algebra.IsSeparable K L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsFractionRing R K
      B : PowerBasis K L
      hint : IsIntegral R B.gen
      z : L
      hz : IsIntegral R z
      hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
      H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
      i : Fin B.dim
      cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).cramer fun i => (Algebra.trace K …
      σ : Equiv.Perm (Fin B.dim)
      x✝¹ : Membership.mem Finset.univ σ
      j : Fin B.dim
      x✝ : Membership.mem Finset.univ j
      hji : Eq j i
      ⊢ Membership.mem Bot.bot ((Algebra.traceMatrix K ⇑B.basis).updateCol i (fun i  …
    -/
  · simp only [updateCol_apply, hji, eq_self_iff_true, PowerBasis.coe_basis]
    /-
      case pos
      K : Type u
      L : Type v
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra K L
      inst✝⁷ : Module.Finite K L
      R : Type z
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R K
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R K L
      inst✝² : Algebra.IsSeparable K L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsFractionRing R K
      B : PowerBasis K L
      hint : IsIntegral R B.gen
      z : L
      hz : IsIntegral R z
      hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
      H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
      i : Fin B.dim
      cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).cramer fun i => (Algebra.trace K …
      σ : Equiv.Perm (Fin B.dim)
      x✝¹ : Membership.mem Finset.univ σ
      j : Fin B.dim
      x✝ : Membership.mem Finset.univ j
      hji : Eq j i
      ⊢ Membership.mem Bot.bot (ite True ((Algebra.trace K L) (HMul.hMul z (HPow.hPo …
    -/
    exact mem_bot.2 (IsIntegrallyClosed.isIntegral_iff.1 <| isIntegral_trace (hz.mul <| hint.pow _))
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u
      L : Type v
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra K L
      inst✝⁷ : Module.Finite K L
      R : Type z
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R K
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R K L
      inst✝² : Algebra.IsSeparable K L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsFractionRing R K
      B : PowerBasis K L
      hint : IsIntegral R B.gen
      z : L
      hz : IsIntegral R z
      hinv : IsUnit (Algebra.traceMatrix K ⇑B.basis).det
      H : Eq (HSMul.hSMul (Algebra.traceMatrix K ⇑B.basis).det ((Algebra.traceMatrix …
      i : Fin B.dim
      cramer : Eq ((Algebra.traceMatrix K ⇑B.basis).cramer fun i => (Algebra.trace K …
      σ : Equiv.Perm (Fin B.dim)
      x✝¹ : Membership.mem Finset.univ σ
      j : Fin B.dim
      x✝ : Membership.mem Finset.univ j
      hji : Not (Eq j i)
      ⊢ Membership.mem Bot.bot ((Algebra.traceMatrix K ⇑B.basis).updateCol i (fun i  …
    -/
  · simp only [updateCol_apply, hji, PowerBasis.coe_basis]
    exact mem_bot.2
      (IsIntegrallyClosed.isIntegral_iff.1 <| isIntegral_trace <| (hint.pow _).mul (hint.pow _))


/-- Two (finite) ℤ-bases have the same discriminant. -/
theorem discr_eq_discr (b : Basis ι ℤ A) (b' : Basis ι ℤ A) :
    Algebra.discr ℤ b = Algebra.discr ℤ b' := by
  /-
    A : Type u
    ι : Type w
    inst✝² : DecidableEq ι
    inst✝¹ : CommRing A
    inst✝ : Fintype ι
    b b' : Basis ι Int A
    ⊢ Eq (Algebra.discr Int ⇑b) (Algebra.discr Int ⇑b')
  -/
  convert Algebra.discr_of_matrix_vecMul b' (b'.toMatrix b)
    /-
      case h.e'_2.h.e'_9
      A : Type u
      ι : Type w
      inst✝² : DecidableEq ι
      inst✝¹ : CommRing A
      inst✝ : Fintype ι
      b b' : Basis ι Int A
      ⊢ Eq (⇑b) (Matrix.vecMul (⇑b') ((b'.toMatrix ⇑b).map ⇑(algebraMap Int A)))
    -/
  · rw [Basis.toMatrix_map_vecMul]
    /-
      🎉 no goals
    -/
  · suffices IsUnit (b'.toMatrix b).det by
      rw [Int.isUnit_iff, ← sq_eq_one_iff] at this
      rw [this, one_mul]
    /-
      case h.e'_3
      A : Type u
      ι : Type w
      inst✝² : DecidableEq ι
      inst✝¹ : CommRing A
      inst✝ : Fintype ι
      b b' : Basis ι Int A
      ⊢ IsUnit (b'.toMatrix ⇑b).det
    -/
    rw [← LinearMap.toMatrix_id_eq_basis_toMatrix b b']
    /-
      case h.e'_3
      A : Type u
      ι : Type w
      inst✝² : DecidableEq ι
      inst✝¹ : CommRing A
      inst✝ : Fintype ι
      b b' : Basis ι Int A
      ⊢ IsUnit ((LinearMap.toMatrix b b') LinearMap.id).det
    -/
    exact LinearEquiv.isUnit_det (LinearEquiv.refl ℤ A) b b'
    /-
      🎉 no goals
    -/


