/-- The infinite adele ring of a number field. -/
def InfiniteAdeleRing := (v : InfinitePlace K) → v.Completion


instance : CommRing (InfiniteAdeleRing K) := Pi.commRing


instance : Inhabited (InfiniteAdeleRing K) := ⟨0⟩


instance [NumberField K] : Nontrivial (InfiniteAdeleRing K) :=
  (inferInstanceAs <| Nonempty (InfinitePlace K)).elim fun w => Pi.nontrivial_at w


instance : TopologicalSpace (InfiniteAdeleRing K) := Pi.topologicalSpace


instance : TopologicalRing (InfiniteAdeleRing K) := Pi.instTopologicalRing


instance : Algebra K (InfiniteAdeleRing K) := Pi.algebra _ _


@[simp]
theorem algebraMap_apply (x : K) (v : InfinitePlace K) :
    algebraMap K (InfiniteAdeleRing K) x v = x := rfl


/-- The infinite adele ring is locally compact. -/
instance locallyCompactSpace [NumberField K] : LocallyCompactSpace (InfiniteAdeleRing K) :=
  Pi.locallyCompactSpace_of_finite


/-- The ring isomorphism between the infinite adele ring of a number field and the
space `ℝ ^ r₁ × ℂ ^ r₂`, where `(r₁, r₂)` is the signature of the number field. -/
abbrev ringEquiv_mixedSpace :
    InfiniteAdeleRing K ≃+* mixedEmbedding.mixedSpace K :=
  RingEquiv.trans
    (RingEquiv.piEquivPiSubtypeProd (fun (v : InfinitePlace K) => IsReal v)
      (fun (v : InfinitePlace K) => v.Completion))
    (RingEquiv.prodCongr
      (RingEquiv.piCongrRight (fun ⟨_, hv⟩ => Completion.ringEquivRealOfIsReal hv))
      (RingEquiv.trans
        (RingEquiv.piCongrRight (fun v => Completion.ringEquivComplexOfIsComplex
          ((not_isReal_iff_isComplex.1 v.2))))
        (RingEquiv.piCongrLeft (fun _ => ℂ) <|
          Equiv.subtypeEquivRight (fun _ => not_isReal_iff_isComplex))))


@[simp]
theorem ringEquiv_mixedSpace_apply (x : InfiniteAdeleRing K) :
    ringEquiv_mixedSpace K x =
      (fun (v : {w : InfinitePlace K // IsReal w}) => extensionEmbeddingOfIsReal v.2 (x v),
       fun (v : {w : InfinitePlace K // IsComplex w}) => extensionEmbedding v.1 (x v)) := rfl


/-- Transfers the embedding of `x ↦ (x)ᵥ` of the number field `K` into its infinite adele
ring to the mixed embedding `x ↦ (φᵢ(x))ᵢ` of `K` into the space `ℝ ^ r₁ × ℂ ^ r₂`, where
`(r₁, r₂)` is the signature of `K` and `φᵢ` are the complex embeddings of `K`. -/
theorem mixedEmbedding_eq_algebraMap_comp {x : K} :
    mixedEmbedding K x = ringEquiv_mixedSpace K (algebraMap K _ x) := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : K
    ⊢ Eq ((NumberField.mixedEmbedding K) x) ((NumberField.InfiniteAdeleRing.ringEq …
  -/
  ext v <;> simp only [ringEquiv_mixedSpace_apply, algebraMap_apply,
    ringEquivRealOfIsReal, ringEquivComplexOfIsComplex, extensionEmbedding,
    extensionEmbeddingOfIsReal, extensionEmbedding_of_comp, RingEquiv.coe_ofBijective,
    RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk, UniformSpace.Completion.extensionHom]
  · rw [UniformSpace.Completion.extension_coe
      (WithAbs.isUniformInducing_of_comp <| v.1.norm_embedding_of_isReal v.2).uniformContinuous x]
    /-
      case fst.h
      K : Type u_1
      inst✝ : Field K
      x : K
      v : Subtype fun w => w.IsReal
      ⊢ Eq (((NumberField.mixedEmbedding K) x).1 v) ((NumberField.InfinitePlace.embe …
    -/
    exact mixedEmbedding.mixedEmbedding_apply_ofIsReal _ _ _
    /-
      🎉 no goals
    -/
  · rw [UniformSpace.Completion.extension_coe
      (WithAbs.isUniformInducing_of_comp <| v.1.norm_embedding_eq).uniformContinuous x]
    /-
      case snd.h
      K : Type u_1
      inst✝ : Field K
      x : K
      v : Subtype fun w => w.IsComplex
      ⊢ Eq (((NumberField.mixedEmbedding K) x).2 v) ((↑v).embedding x)
    -/
    exact mixedEmbedding.mixedEmbedding_apply_ofIsComplex _ _ _
    /-
      🎉 no goals
    -/


/-- The adele ring of a number field. -/
def AdeleRing := InfiniteAdeleRing K × FiniteAdeleRing (𝓞 K) K


instance : CommRing (AdeleRing K) := Prod.instCommRing


instance : Inhabited (AdeleRing K) := ⟨0⟩


instance : TopologicalSpace (AdeleRing K) := instTopologicalSpaceProd


instance : TopologicalRing (AdeleRing K) := instTopologicalRingProd


instance : Algebra K (AdeleRing K) := Prod.algebra _ _ _


@[simp]
theorem algebraMap_fst_apply (x : K) (v : InfinitePlace K) :
    (algebraMap K (AdeleRing K) x).1 v = x := rfl


@[simp]
theorem algebraMap_snd_apply (x : K) (v : HeightOneSpectrum (𝓞 K)) :
    (algebraMap K (AdeleRing K) x).2 v = x := rfl


theorem algebraMap_injective : Function.Injective (algebraMap K (AdeleRing K)) :=
  fun _ _ hxy => (algebraMap K _).injective (Prod.ext_iff.1 hxy).1


/-- The subgroup of principal adeles `(x)ᵥ` where `x ∈ K`. -/
abbrev principalSubgroup : AddSubgroup (AdeleRing K) := (algebraMap K _).range.toAddSubgroup


