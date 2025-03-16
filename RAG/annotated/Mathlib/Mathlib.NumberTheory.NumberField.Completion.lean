/-- The completion of a number field at an infinite place. -/
abbrev Completion := v.1.Completion


@[deprecated (since := "2024-12-01")] alias completion := Completion


instance : NormedField v.Completion :=
  letI := (WithAbs.isUniformInducing_of_comp v.norm_embedding_eq).completableTopField
  UniformSpace.Completion.instNormedFieldOfCompletableTopField (WithAbs v.1)


lemma norm_coe (x : WithAbs v.1) :
    ‖(x : v.Completion)‖ = v (WithAbs.equiv v.1 x) :=
  UniformSpace.Completion.norm_coe x


instance : Algebra K v.Completion :=
  inferInstanceAs <| Algebra (WithAbs v.1) v.1.Completion


/-- The coercion from the rationals to its completion along an infinite place is `Rat.cast`. -/
lemma WithAbs.ratCast_equiv (v : InfinitePlace ℚ) (x : WithAbs v.1) :
    Rat.cast (WithAbs.equiv _ x) = (x : v.Completion) :=
  (eq_ratCast (UniformSpace.Completion.coeRingHom.comp
    (WithAbs.ringEquiv v.1).symm.toRingHom) x).symm


lemma Rat.norm_infinitePlace_completion (v : InfinitePlace ℚ) (x : ℚ) :
    ‖(x : v.Completion)‖ = |x| := by
  rw [← (WithAbs.equiv v.1).apply_symm_apply x, WithAbs.ratCast_equiv,
    norm_coe, (WithAbs.equiv v.1).apply_symm_apply,
    Rat.infinitePlace_apply]


/-- The completion of a number field at an infinite place is locally compact. -/
instance locallyCompactSpace : LocallyCompactSpace (v.Completion) :=
  AbsoluteValue.Completion.locallyCompactSpace v.norm_embedding_eq


/-- The embedding associated to an infinite place extended to an embedding `v.Completion →+* ℂ`. -/
def extensionEmbedding : v.Completion →+* ℂ := extensionEmbedding_of_comp v.norm_embedding_eq


/-- The embedding `K →+* ℝ` associated to a real infinite place extended to `v.Completion →+* ℝ`. -/
def extensionEmbeddingOfIsReal {v : InfinitePlace K} (hv : IsReal v) : v.Completion →+* ℝ :=
  extensionEmbedding_of_comp <| v.norm_embedding_of_isReal hv


@[deprecated (since := "2024-12-07")]
noncomputable alias extensionEmbedding_of_isReal := extensionEmbeddingOfIsReal


@[simp]
theorem extensionEmbedding_coe (x : K) : extensionEmbedding v x = v.embedding x :=
  extensionEmbedding_of_comp_coe v.norm_embedding_eq x


@[simp]
theorem extensionEmbedding_of_isReal_coe {v : InfinitePlace K} (hv : IsReal v) (x : K) :
    extensionEmbeddingOfIsReal hv x = embedding_of_isReal hv x :=
  extensionEmbedding_of_comp_coe (v.norm_embedding_of_isReal hv) x


/-- The embedding `v.Completion →+* ℂ` is an isometry. -/
theorem isometry_extensionEmbedding : Isometry (extensionEmbedding v) :=
  Isometry.of_dist_eq (extensionEmbedding_dist_eq_of_comp v.norm_embedding_eq)


/-- The embedding `v.Completion →+* ℝ` at a real infinite place is an isometry. -/
theorem isometry_extensionEmbedding_of_isReal {v : InfinitePlace K} (hv : IsReal v) :
    Isometry (extensionEmbeddingOfIsReal hv) :=
  Isometry.of_dist_eq (extensionEmbedding_dist_eq_of_comp <| v.norm_embedding_of_isReal hv)


/-- The embedding `v.Completion →+* ℂ` has closed image inside `ℂ`. -/
theorem isClosed_image_extensionEmbedding : IsClosed (Set.range (extensionEmbedding v)) :=
  (isClosedEmbedding_extensionEmbedding_of_comp v.norm_embedding_eq).isClosed_range


/-- The embedding `v.Completion →+* ℝ` associated to a real infinite place has closed image
inside `ℝ`. -/
theorem isClosed_image_extensionEmbedding_of_isReal {v : InfinitePlace K} (hv : IsReal v) :
    IsClosed (Set.range (extensionEmbeddingOfIsReal hv)) :=
  (isClosedEmbedding_extensionEmbedding_of_comp <| v.norm_embedding_of_isReal hv).isClosed_range


theorem subfield_ne_real_of_isComplex {v : InfinitePlace K} (hv : IsComplex v) :
    (extensionEmbedding v).fieldRange ≠ Complex.ofRealHom.fieldRange := by
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : v.IsComplex
    ⊢ Ne (NumberField.InfinitePlace.Completion.extensionEmbedding v).fieldRange Co …
  -/
  contrapose! hv
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : Eq (NumberField.InfinitePlace.Completion.extensionEmbedding v).fieldRange …
    ⊢ Not v.IsComplex
  -/
  simp only [not_isComplex_iff_isReal, isReal_iff]
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : Eq (NumberField.InfinitePlace.Completion.extensionEmbedding v).fieldRange …
    ⊢ NumberField.ComplexEmbedding.IsReal v.embedding
  -/
  ext x
  /-
    case a
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : Eq (NumberField.InfinitePlace.Completion.extensionEmbedding v).fieldRange …
    x : K
    ⊢ Eq ((Star.star v.embedding) x) (v.embedding x)
  -/
  obtain ⟨r, hr⟩ := hv ▸ extensionEmbedding_coe v x ▸ RingHom.mem_fieldRange_self _ _
  /-
    case a.intro
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : Eq (NumberField.InfinitePlace.Completion.extensionEmbedding v).fieldRange …
    x : K
    r : Real
    hr : Eq (Complex.ofRealHom r) (v.embedding x)
    ⊢ Eq ((Star.star v.embedding) x) (v.embedding x)
  -/
  simp only [ComplexEmbedding.conjugate_coe_eq, ← hr, Complex.ofRealHom_eq_coe, Complex.conj_ofReal]
  /-
    🎉 no goals
  -/


/-- If `v` is a complex infinite place, then the embedding `v.Completion →+* ℂ` is surjective. -/
theorem surjective_extensionEmbedding_of_isComplex {v : InfinitePlace K} (hv : IsComplex v) :
    Function.Surjective (extensionEmbedding v) := by
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : v.IsComplex
    ⊢ Function.Surjective ⇑(NumberField.InfinitePlace.Completion.extensionEmbeddin …
  -/
  rw [← RingHom.fieldRange_eq_top_iff]
  exact (Complex.subfield_eq_of_closed <| isClosed_image_extensionEmbedding v).resolve_left <|
    subfield_ne_real_of_isComplex hv


/-- If `v` is a complex infinite place, then the embedding `v.Completion →+* ℂ` is bijective. -/
theorem bijective_extensionEmbedding_of_isComplex {v : InfinitePlace K} (hv : IsComplex v) :
    Function.Bijective (extensionEmbedding v) :=
  ⟨(extensionEmbedding v).injective, surjective_extensionEmbedding_of_isComplex hv⟩


/-- The ring isomorphism `v.Completion ≃+* ℂ`, when `v` is complex, given by the bijection
`v.Completion →+* ℂ`. -/
def ringEquivComplexOfIsComplex {v : InfinitePlace K} (hv : IsComplex v) :
    v.Completion ≃+* ℂ :=
  RingEquiv.ofBijective _ (bijective_extensionEmbedding_of_isComplex hv)


@[deprecated (since := "2024-12-07")]
noncomputable alias ringEquiv_complex_of_isComplex := ringEquivComplexOfIsComplex


/-- If the infinite place `v` is complex, then `v.Completion` is isometric to `ℂ`. -/
def isometryEquivComplexOfIsComplex {v : InfinitePlace K} (hv : IsComplex v) :
    v.Completion ≃ᵢ ℂ where
  toEquiv := ringEquivComplexOfIsComplex hv
  isometry_toFun := isometry_extensionEmbedding v


@[deprecated (since := "2024-12-07")]
noncomputable alias isometryEquiv_complex_of_isComplex := isometryEquivComplexOfIsComplex


/-- If `v` is a real infinite place, then the embedding `v.Completion →+* ℝ` is surjective. -/
theorem surjective_extensionEmbedding_of_isReal {v : InfinitePlace K} (hv : IsReal v) :
    Function.Surjective (extensionEmbeddingOfIsReal hv) := by
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : v.IsReal
    ⊢ Function.Surjective ⇑(NumberField.InfinitePlace.Completion.extensionEmbeddin …
  -/
  rw [← RingHom.fieldRange_eq_top_iff, ← Real.subfield_eq_of_closed]
  /-
    K : Type u_1
    inst✝ : Field K
    v : NumberField.InfinitePlace K
    hv : v.IsReal
    ⊢ IsClosed ↑(NumberField.InfinitePlace.Completion.extensionEmbeddingOfIsReal h …
  -/
  exact isClosed_image_extensionEmbedding_of_isReal hv
  /-
    🎉 no goals
  -/


/-- If `v` is a real infinite place, then the embedding `v.Completion →+* ℝ` is bijective. -/
theorem bijective_extensionEmbedding_of_isReal {v : InfinitePlace K} (hv : IsReal v) :
    Function.Bijective (extensionEmbeddingOfIsReal hv) :=
  ⟨(extensionEmbeddingOfIsReal hv).injective, surjective_extensionEmbedding_of_isReal hv⟩


/-- The ring isomorphism `v.Completion ≃+* ℝ`, when `v` is real, given by the bijection
`v.Completion →+* ℝ`. -/
def ringEquivRealOfIsReal {v : InfinitePlace K} (hv : IsReal v) : v.Completion ≃+* ℝ :=
  RingEquiv.ofBijective _ (bijective_extensionEmbedding_of_isReal hv)


@[deprecated (since := "2024-12-07")]
noncomputable alias ringEquiv_real_of_isReal := ringEquivRealOfIsReal


/-- If the infinite place `v` is real, then `v.Completion` is isometric to `ℝ`. -/
def isometryEquivRealOfIsReal {v : InfinitePlace K} (hv : IsReal v) : v.Completion ≃ᵢ ℝ where
  toEquiv := ringEquivRealOfIsReal hv
  isometry_toFun := isometry_extensionEmbedding_of_isReal hv


@[deprecated (since := "2024-12-07")]
noncomputable alias isometryEquiv_real_of_isReal := isometryEquivRealOfIsReal


