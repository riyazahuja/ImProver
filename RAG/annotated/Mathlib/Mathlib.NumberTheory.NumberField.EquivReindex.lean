/-- An equivalence between the set of embeddings of `K` into `ℂ` and the
  index set of the chosen basis of the ring of integers of `K`. -/
abbrev equivReindex : (K →+* ℂ) ≃ (ChooseBasisIndex ℤ (𝓞 K)) :=
    Fintype.equivOfCardEq <|
     /-
       K : Type u_1
       inst✝¹ : Field K
       inst✝ : NumberField K
       ⊢ Eq (Fintype.card (RingHom K Complex)) (Fintype.card (Module.Free.ChooseBasis …
     -/
  by rw [Embeddings.card, ← finrank_eq_card_chooseBasisIndex, RingOfIntegers.rank]
     /-
       🎉 no goals
     -/


/-- The basis matrix for the embeddings of `K` into `ℂ`. This matrix is formed by
  taking the lattice basis vectors of `K` and reindexing them according to the
  equivalence `equivReindex`, then transposing the resulting matrix. -/
abbrev basisMatrix : Matrix (K →+* ℂ) (K →+* ℂ) ℂ :=
  (Matrix.of fun i ↦ latticeBasis K (equivReindex K i))


theorem det_of_basisMatrix_non_zero [DecidableEq (K →+* ℂ)] : (basisMatrix K).det ≠ 0 := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    ⊢ Ne (NumberField.basisMatrix K).det 0
  -/
  let e : (K →+* ℂ) ≃ ChooseBasisIndex ℤ (𝓞 K) := equivReindex K
  let N := Algebra.embeddingsMatrixReindex ℚ ℂ (fun i => integralBasis K (e i))
    RingHom.equivRatAlgHom
  rw [show (basisMatrix K) = N by
    ext:2; simp only [N, transpose_apply, latticeBasis_apply, integralBasis_apply,
    of_apply, apply_at]; rfl, ← pow_ne_zero_iff two_ne_zero]
  convert (map_ne_zero_iff _ (algebraMap ℚ ℂ).injective).mpr
    (Algebra.discr_not_zero_of_basis ℚ (integralBasis K))
  /-
    case h.e'_2
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    e : Equiv (RingHom K Complex) (Module.Free.ChooseBasisIndex Int (NumberField.R …
    N : Matrix (RingHom K Complex) (RingHom K Complex) Complex := Algebra.embeddin …
    ⊢ Eq (HPow.hPow N.det 2) ((algebraMap Rat Complex) (Algebra.discr Rat ⇑(Number …
  -/
  rw [← Algebra.discr_reindex ℚ (integralBasis K) e.symm]
  exact (Algebra.discr_eq_det_embeddingsMatrixReindex_pow_two ℚ ℂ
    (fun _ => integralBasis K (e _)) RingHom.equivRatAlgHom).symm


instance [DecidableEq (K →+* ℂ)] : Invertible (basisMatrix K) := invertibleOfIsUnitDet _
    (Ne.isUnit (det_of_basisMatrix_non_zero K))


theorem canonicalEmbedding_eq_basisMatrix_mulVec (α : K) :
    canonicalEmbedding K α = (basisMatrix K).transpose.mulVec
      (fun i ↦ (((integralBasis K).reindex (equivReindex K).symm).repr α i : ℂ)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : K
    ⊢ Eq ((NumberField.canonicalEmbedding K) α) ((NumberField.basisMatrix K).trans …
  -/
  ext i
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    α : K
    i : RingHom K Complex
    ⊢ Eq ((NumberField.canonicalEmbedding K) α i) ((NumberField.basisMatrix K).tra …
  -/
  rw [← (latticeBasis K).sum_repr (canonicalEmbedding K α), ← Equiv.sum_comp (equivReindex K)]
  simp only [canonicalEmbedding.integralBasis_repr_apply, mulVec, dotProduct,
    transpose_apply, of_apply, Fintype.sum_apply, mul_comm, Basis.repr_reindex,
    Finsupp.mapDomain_equiv_apply, Equiv.symm_symm, Pi.smul_apply, smul_eq_mul]


theorem inverse_basisMatrix_mulVec_eq_repr [DecidableEq (K →+* ℂ)] (α : 𝓞 K) :
    ∀ i, ((basisMatrix K).transpose)⁻¹.mulVec (fun j =>
      canonicalEmbedding K (algebraMap (𝓞 K) K α) j) i =
      ((integralBasis K).reindex (equivReindex K).symm).repr α i := fun i => by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : DecidableEq (RingHom K Complex)
    α : NumberField.RingOfIntegers K
    i : RingHom K Complex
    ⊢ Eq ((Inv.inv (NumberField.basisMatrix K).transpose).mulVec (fun j => (Number …
  -/
  rw [inv_mulVec_eq_vec (canonicalEmbedding_eq_basisMatrix_mulVec ((algebraMap (𝓞 K) K) α))]
  /-
    🎉 no goals
  -/


