theorem isEmbedding_sigmaMk_comp [Nonempty X] :
    IsEmbedding (fun g : Σ i, C(X, Y i) ↦ (sigmaMk g.1).comp g.2) where
  toIsInducing := inducing_sigma.2
    ⟨fun i ↦ (sigmaMk i).isInducing_postcomp IsEmbedding.sigmaMk.isInducing, fun i ↦
      let ⟨x⟩ := ‹Nonempty X›
      ⟨_, (isOpen_sigma_fst_preimage {i}).preimage (continuous_eval_const x), fun _ ↦ Iff.rfl⟩⟩
  injective := by
    /-
      X : Type u_1
      ι : Type u_2
      Y : ι → Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : (i : ι) → TopologicalSpace (Y i)
      inst✝ : Nonempty X
      ⊢ Function.Injective fun g => (ContinuousMap.sigmaMk g.fst).comp g.snd
    -/
    rintro ⟨i, g⟩ ⟨i', g'⟩ h
    obtain ⟨rfl, hg⟩ : i = i' ∧ HEq (⇑g) (⇑g') :=
      Function.eq_of_sigmaMk_comp <| congr_arg DFunLike.coe h
    /-
      case mk.mk.intro
      X : Type u_1
      ι : Type u_2
      Y : ι → Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : (i : ι) → TopologicalSpace (Y i)
      inst✝ : Nonempty X
      i : ι
      g g' : ContinuousMap X (Y i)
      h : Eq ((fun g => (ContinuousMap.sigmaMk g.fst).comp g.snd) ⟨i, g⟩) ((fun g => …
      hg : HEq ⇑g ⇑g'
      ⊢ Eq ⟨i, g⟩ ⟨i, g'⟩
    -/
    simpa using hg
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias embedding_sigmaMk_comp := isEmbedding_sigmaMk_comp


/-- Every continuous map from a connected topological space to the disjoint union of a family of
topological spaces is a composition of the embedding `ContinuousMap.sigmMk i : C(Y i, Σ i, Y i)` for
some `i` and a continuous map `g : C(X, Y i)`. See also `Continuous.exists_lift_sigma` for a version
with unbundled functions and `ContinuousMap.sigmaCodHomeomorph` for a homeomorphism defined using
this fact. -/
theorem exists_lift_sigma (f : C(X, Σ i, Y i)) : ∃ i g, f = (sigmaMk i).comp g :=
  let ⟨i, g, hg, hfg⟩ := (map_continuous f).exists_lift_sigma
  ⟨i, ⟨g, hg⟩, DFunLike.ext' hfg⟩


/-- Homeomorphism between the type `C(X, Σ i, Y i)` of continuous maps from a connected topological
space to the disjoint union of a family of topological spaces and the disjoint union of the types of
continuous maps `C(X, Y i)`.

The inverse map sends `⟨i, g⟩` to `ContinuousMap.comp (ContinuousMap.sigmaMk i) g`. -/
@[simps! symm_apply]
def sigmaCodHomeomorph : C(X, Σ i, Y i) ≃ₜ Σ i, C(X, Y i) :=
  .symm <| Equiv.toHomeomorphOfIsInducing
    (.ofBijective _ ⟨isEmbedding_sigmaMk_comp.injective, fun f ↦
      let ⟨i, g, hg⟩ := f.exists_lift_sigma; ⟨⟨i, g⟩, hg.symm⟩⟩)
    isEmbedding_sigmaMk_comp.isInducing


