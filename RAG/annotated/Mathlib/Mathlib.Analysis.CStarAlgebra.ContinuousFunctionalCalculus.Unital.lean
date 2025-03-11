/-- A star `R`-algebra `A` has a *continuous functional calculus* for elements satisfying the
property `p : A → Prop` if

+ for every such element `a : A` there is a star algebra homomorphism
  `cfcHom : C(spectrum R a, R) →⋆ₐ[R] A` sending the (restriction of) the identity map to `a`.
+ `cfcHom` is a closed embedding for which the spectrum of the image of function `f` is its range.
+ `cfcHom` preserves the property `p`.
+ `p 0` is true, which ensures among other things that `p ≠ fun _ ↦ False`.

The property `p` is marked as an `outParam` so that the user need not specify it. In practice,

+ for `R := ℂ`, we choose `p := IsStarNormal`,
+ for `R := ℝ`, we choose `p := IsSelfAdjoint`,
+ for `R := ℝ≥0`, we choose `p := (0 ≤ ·)`.

Instead of directly providing the data we opt instead for a `Prop` class. In all relevant cases,
the continuous functional calculus is uniquely determined, and utilizing this approach
prevents diamonds or problems arising from multiple instances. -/
class ContinuousFunctionalCalculus (R : Type*) {A : Type*} (p : outParam (A → Prop))
    [CommSemiring R] [StarRing R] [MetricSpace R] [TopologicalSemiring R] [ContinuousStar R]
    [Ring A] [StarRing A] [TopologicalSpace A] [Algebra R A] : Prop where
  predicate_zero : p 0
  [compactSpace_spectrum (a : A) : CompactSpace (spectrum R a)]
  spectrum_nonempty [Nontrivial A] (a : A) (ha : p a) : (spectrum R a).Nonempty
  exists_cfc_of_predicate : ∀ a, p a → ∃ φ : C(spectrum R a, R) →⋆ₐ[R] A,
    IsClosedEmbedding φ ∧ φ ((ContinuousMap.id R).restrict <| spectrum R a) = a ∧
      (∀ f, spectrum R (φ f) = Set.range f) ∧ ∀ f, p (φ f)

-- this instance should not be activated everywhere but it is useful when developing generic API
-- for the continuous functional calculus

/-- A class guaranteeing that the continuous functional calculus is uniquely determined by the
properties that it is a continuous star algebra homomorphism mapping the (restriction of) the
identity to `a`. This is the necessary tool used to establish `cfcHom_comp` and the more common
variant `cfc_comp`.

This class has instances, which can be found in
`Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Unique`, in each of the common cases
`ℂ`, `ℝ` and `ℝ≥0` as a consequence of the Stone-Weierstrass theorem.

This class is separate from `ContinuousFunctionalCalculus` primarily because we will later use
`SpectrumRestricts` to derive an instance of `ContinuousFunctionalCalculus` on a scalar subring
from one on a larger ring (i.e., to go from a continuous functional calculus over `ℂ` for normal
elements to one over `ℝ` for selfadjoint elements), and proving this additional property is
preserved would be burdensome or impossible. -/
class UniqueContinuousFunctionalCalculus (R A : Type*) [CommSemiring R] [StarRing R]
    [MetricSpace R] [TopologicalSemiring R] [ContinuousStar R] [Ring A] [StarRing A]
    [TopologicalSpace A] [Algebra R A] : Prop where
  eq_of_continuous_of_map_id (s : Set R) [CompactSpace s]
    (φ ψ : C(s, R) →⋆ₐ[R] A) (hφ : Continuous φ) (hψ : Continuous ψ)
    (h : φ (.restrict s <| .id R) = ψ (.restrict s <| .id R)) :
    φ = ψ
  compactSpace_spectrum (a : A) : CompactSpace (spectrum R a)


include instCFC in
lemma ContinuousFunctionalCalculus.isCompact_spectrum (a : A) :
    IsCompact (spectrum R a) :=
  isCompact_iff_compactSpace.mpr inferInstance


lemma StarAlgHom.ext_continuousMap [UniqueContinuousFunctionalCalculus R A]
    (a : A) (φ ψ : C(spectrum R a, R) →⋆ₐ[R] A) (hφ : Continuous φ) (hψ : Continuous ψ)
    (h : φ (.restrict (spectrum R a) <| .id R) = ψ (.restrict (spectrum R a) <| .id R)) :
    φ = ψ :=
  have := UniqueContinuousFunctionalCalculus.compactSpace_spectrum (R := R) a
  UniqueContinuousFunctionalCalculus.eq_of_continuous_of_map_id (spectrum R a) φ ψ hφ hψ h


/-- The star algebra homomorphism underlying a instance of the continuous functional calculus;
a version for continuous functions on the spectrum.

In this case, the user must supply the fact that `a` satisfies the predicate `p`, for otherwise it
may be the case that no star algebra homomorphism exists. For instance if `R := ℝ` and `a` is an
element whose spectrum (in `ℂ`) is disjoint from `ℝ`, then `spectrum ℝ a = ∅` and so there can be
no star algebra homomorphism between these spaces.

While `ContinuousFunctionalCalculus` is stated in terms of these homomorphisms, in practice the
user should instead prefer `cfc` over `cfcHom`.
-/
noncomputable def cfcHom : C(spectrum R a, R) →⋆ₐ[R] A :=
  (ContinuousFunctionalCalculus.exists_cfc_of_predicate a ha).choose


lemma cfcHom_isClosedEmbedding :
    IsClosedEmbedding <| (cfcHom ha : C(spectrum R a, R) →⋆ₐ[R] A) :=
  (ContinuousFunctionalCalculus.exists_cfc_of_predicate a ha).choose_spec.1


@[deprecated (since := "2024-10-20")]
alias cfcHom_closedEmbedding := cfcHom_isClosedEmbedding


@[fun_prop]
lemma cfcHom_continuous : Continuous (cfcHom ha : C(spectrum R a, R) →⋆ₐ[R] A) :=
  cfcHom_isClosedEmbedding ha |>.continuous


lemma cfcHom_id :
    cfcHom ha ((ContinuousMap.id R).restrict <| spectrum R a) = a :=
  (ContinuousFunctionalCalculus.exists_cfc_of_predicate a ha).choose_spec.2.1


/-- The **spectral mapping theorem** for the continuous functional calculus. -/
lemma cfcHom_map_spectrum (f : C(spectrum R a, R)) :
    spectrum R (cfcHom ha f) = Set.range f :=
  (ContinuousFunctionalCalculus.exists_cfc_of_predicate a ha).choose_spec.2.2.1 f


lemma cfcHom_predicate (f : C(spectrum R a, R)) :
    p (cfcHom ha f) :=
  (ContinuousFunctionalCalculus.exists_cfc_of_predicate a ha).choose_spec.2.2.2 f


lemma cfcHom_eq_of_continuous_of_map_id [UniqueContinuousFunctionalCalculus R A]
    (φ : C(spectrum R a, R) →⋆ₐ[R] A) (hφ₁ : Continuous φ)
    (hφ₂ : φ (.restrict (spectrum R a) <| .id R) = a) : cfcHom ha = φ :=
  (cfcHom ha).ext_continuousMap a φ (cfcHom_isClosedEmbedding ha).continuous hφ₁ <| by
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R a)) R) A
      hφ₁ : Continuous ⇑φ
      hφ₂ : Eq (φ (ContinuousMap.restrict (spectrum R a) (ContinuousMap.id R))) a
      ⊢ Eq ((cfcHom ha) (ContinuousMap.restrict (spectrum R a) (ContinuousMap.id R)) …
    -/
    rw [cfcHom_id ha, hφ₂]
    /-
      🎉 no goals
    -/


theorem cfcHom_comp [UniqueContinuousFunctionalCalculus R A] (f : C(spectrum R a, R))
    (f' : C(spectrum R a, spectrum R (cfcHom ha f)))
    (hff' : ∀ x, f x = f' x) (g : C(spectrum R (cfcHom ha f), R)) :
    cfcHom ha (g.comp f') = cfcHom (cfcHom_predicate ha f) g := by
  let φ : C(spectrum R (cfcHom ha f), R) →⋆ₐ[R] A :=
    (cfcHom ha).comp <| ContinuousMap.compStarAlgHom' R R f'
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : p a
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : ContinuousMap (↑(spectrum R a)) R
    f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
    hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
    g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
    φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
    ⊢ Eq ((cfcHom ha) (g.comp f')) ((cfcHom ⋯) g)
  -/
  suffices cfcHom (cfcHom_predicate ha f) = φ from DFunLike.congr_fun this.symm g
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : p a
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : ContinuousMap (↑(spectrum R a)) R
    f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
    hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
    g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
    φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
    ⊢ Eq (cfcHom ⋯) φ
  -/
  refine cfcHom_eq_of_continuous_of_map_id (cfcHom_predicate ha f) φ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap (↑(spectrum R a)) R
      f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
      hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
      g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
      ⊢ Continuous ⇑φ
    -/
  · exact (cfcHom_isClosedEmbedding ha).continuous.comp f'.continuous_precomp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap (↑(spectrum R a)) R
      f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
      hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
      g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
      ⊢ Eq (φ (ContinuousMap.restrict (spectrum R ((cfcHom ha) f)) (ContinuousMap.id …
    -/
  · simp only [φ, StarAlgHom.comp_apply, ContinuousMap.compStarAlgHom'_apply]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap (↑(spectrum R a)) R
      f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
      hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
      g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
      ⊢ Eq ((cfcHom ha) ((ContinuousMap.restrict (spectrum R ((cfcHom ha) f)) (Conti …
    -/
    congr
    /-
      case refine_2.h.e_6.h
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap (↑(spectrum R a)) R
      f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
      hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
      g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
      ⊢ Eq ((ContinuousMap.restrict (spectrum R ((cfcHom ha) f)) (ContinuousMap.id R …
    -/
    ext x
    /-
      case refine_2.h.e_6.h.h
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap (↑(spectrum R a)) R
      f' : ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ha) f))
      hff' : ∀ (x : ↑(spectrum R a)), Eq (f x) ↑(f' x)
      g : ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R
      φ : StarAlgHom R (ContinuousMap (↑(spectrum R ((cfcHom ha) f))) R) A := (cfcHo …
      x : ↑(spectrum R a)
      ⊢ Eq (((ContinuousMap.restrict (spectrum R ((cfcHom ha) f)) (ContinuousMap.id  …
    -/
    simp [hff']
    /-
      🎉 no goals
    -/


/-- `cfcHom` bundled as a continuous linear map. -/
@[simps apply]
noncomputable def cfcL {a : A} (ha : p a) : C(spectrum R a, R) →L[R] A :=
  { cfcHom ha with
    toFun := cfcHom ha
    map_smul' := map_smul _
    cont := (cfcHom_isClosedEmbedding ha).continuous }


open scoped Classical in
/-- This is the *continuous functional calculus* of an element `a : A` applied to bare functions.
When either `a` does not satisfy the predicate `p` (i.e., `a` is not `IsStarNormal`,
`IsSelfAdjoint`, or `0 ≤ a` when `R` is `ℂ`, `ℝ`, or `ℝ≥0`, respectively), or when `f : R → R` is
not continuous on the spectrum of `a`, then `cfc f a` returns the junk value `0`.

This is the primary declaration intended for widespread use of the continuous functional calculus,
and all the API applies to this declaration. For more information, see the module documentation
for `Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Unital`. -/
noncomputable irreducible_def cfc (f : R → R) (a : A) : A :=
  if h : p a ∧ ContinuousOn f (spectrum R a)
    then cfcHom h.1 ⟨_, h.2.restrict⟩
    else 0


variable (f g : R → R) (a : A) (ha : p a := by cfc_tac)

variable (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)

variable (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac)


lemma cfc_apply : cfc f a = cfcHom (a := a) ha ⟨_, hf.restrict⟩ := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Eq (cfc f a) ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_t …
  -/
  rw [cfc_def, dif_pos ⟨ha, hf⟩]
  /-
    🎉 no goals
  -/


lemma cfc_apply_pi {ι : Type*} (f : ι → R → R) (a : A) (ha : p a := by cfc_tac)
    (hf : ∀ i, ContinuousOn (f i) (spectrum R a) := by cfc_cont_tac) :
    (fun i => cfc (f i) a) = (fun i => cfcHom (a := a) ha ⟨_, (hf i).restrict⟩) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    ι : Type u_3
    f : ι → R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (∀ (i : ι), ContinuousOn (f i) (spectrum R a)) _auto✝
    ⊢ Eq (fun i => cfc (f i) a) fun i => (cfcHom ha) { toFun := (spectrum R a).res …
  -/
  ext i
  /-
    case h
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    ι : Type u_3
    f : ι → R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (∀ (i : ι), ContinuousOn (f i) (spectrum R a)) _auto✝
    i : ι
    ⊢ Eq (cfc (f i) a) ((cfcHom ha) { toFun := (spectrum R a).restrict (f i), cont …
  -/
  simp only [cfc_apply (f i) a ha (hf i)]
  /-
    🎉 no goals
  -/


lemma cfc_apply_of_not_and {f : R → R} (a : A) (ha : ¬ (p a ∧ ContinuousOn f (spectrum R a))) :
    cfc f a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : Not (And (p a) (ContinuousOn f (spectrum R a)))
    ⊢ Eq (cfc f a) 0
  -/
  rw [cfc_def, dif_neg ha]
  /-
    🎉 no goals
  -/


lemma cfc_apply_of_not_predicate {f : R → R} (a : A) (ha : ¬ p a) :
    cfc f a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : Not (p a)
    ⊢ Eq (cfc f a) 0
  -/
  rw [cfc_def, dif_neg (not_and_of_not_left _ ha)]
  /-
    🎉 no goals
  -/


lemma cfc_apply_of_not_continuousOn {f : R → R} (a : A) (hf : ¬ ContinuousOn f (spectrum R a)) :
    cfc f a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    hf : Not (ContinuousOn f (spectrum R a))
    ⊢ Eq (cfc f a) 0
  -/
  rw [cfc_def, dif_neg (not_and_of_not_right _ hf)]
  /-
    🎉 no goals
  -/


lemma cfcHom_eq_cfc_extend {a : A} (g : R → R) (ha : p a) (f : C(spectrum R a, R)) :
    cfcHom ha f = cfc (Function.extend Subtype.val f g) a := by
  have h : f = (spectrum R a).restrict (Function.extend Subtype.val f g) := by
    ext; simp [Subtype.val_injective.extend_apply]
  have hg : ContinuousOn (Function.extend Subtype.val f g) (spectrum R a) :=
    continuousOn_iff_continuous_restrict.mpr <| h ▸ map_continuous f
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    g : R → R
    ha : p a
    f : ContinuousMap (↑(spectrum R a)) R
    h : Eq (⇑f) ((spectrum R a).restrict (Function.extend Subtype.val (⇑f) g))
    hg : ContinuousOn (Function.extend Subtype.val (⇑f) g) (spectrum R a)
    ⊢ Eq ((cfcHom ha) f) (cfc (Function.extend Subtype.val (⇑f) g) a)
  -/
  rw [cfc_apply ..]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    g : R → R
    ha : p a
    f : ContinuousMap (↑(spectrum R a)) R
    h : Eq (⇑f) ((spectrum R a).restrict (Function.extend Subtype.val (⇑f) g))
    hg : ContinuousOn (Function.extend Subtype.val (⇑f) g) (spectrum R a)
    ⊢ Eq ((cfcHom ha) f) ((cfcHom ha) { toFun := (spectrum R a).restrict (Function …
  -/
  congr!
  /-
    🎉 no goals
  -/


lemma cfc_eq_cfcL {a : A} {f : R → R} (ha : p a) (hf : ContinuousOn f (spectrum R a)) :
    cfc f a = cfcL ha ⟨_, hf.restrict⟩ := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    f : R → R
    ha : p a
    hf : ContinuousOn f (spectrum R a)
    ⊢ Eq (cfc f a) ((cfcL ha) { toFun := (spectrum R a).restrict f, continuous_toF …
  -/
  rw [cfc_def, dif_pos ⟨ha, hf⟩, cfcL_apply]
  /-
    🎉 no goals
  -/


lemma cfc_cases (P : A → Prop) (a : A) (f : R → R) (h₀ : P 0)
    (haf : (hf : ContinuousOn f (spectrum R a)) → (ha : p a) → P (cfcHom ha ⟨_, hf.restrict⟩)) :
    P (cfc f a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    P : A → Prop
    a : A
    f : R → R
    h₀ : P 0
    haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
    ⊢ P (cfc f a)
  -/
  by_cases h : p a ∧ ContinuousOn f (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      P : A → Prop
      a : A
      f : R → R
      h₀ : P 0
      haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
      h : And (p a) (ContinuousOn f (spectrum R a))
      ⊢ P (cfc f a)
    -/
  · rw [cfc_apply f a h.1 h.2]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      P : A → Prop
      a : A
      f : R → R
      h₀ : P 0
      haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
      h : And (p a) (ContinuousOn f (spectrum R a))
      ⊢ P ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, continuous_toFun := ⋯ })
    -/
    exact haf h.2 h.1
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      P : A → Prop
      a : A
      f : R → R
      h₀ : P 0
      haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
      h : Not (And (p a) (ContinuousOn f (spectrum R a)))
      ⊢ P (cfc f a)
    -/
  · simp only [not_and_or] at h
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      P : A → Prop
      a : A
      f : R → R
      h₀ : P 0
      haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
      h : Or (Not (p a)) (Not (ContinuousOn f (spectrum R a)))
      ⊢ P (cfc f a)
    -/
    obtain (h | h) := h
      /-
        case neg.inl
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        P : A → Prop
        a : A
        f : R → R
        h₀ : P 0
        haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
        h : Not (p a)
        ⊢ P (cfc f a)
      -/
    · rwa [cfc_apply_of_not_predicate _ h]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        P : A → Prop
        a : A
        f : R → R
        h₀ : P 0
        haf : ∀ (hf : ContinuousOn f (spectrum R a)) (ha : p a), P ((cfcHom ha) { toFu …
        h : Not (ContinuousOn f (spectrum R a))
        ⊢ P (cfc f a)
      -/
    · rwa [cfc_apply_of_not_continuousOn _ h]
      /-
        🎉 no goals
      -/


lemma cfc_commute_cfc (f g : R → R) (a : A) : Commute (cfc f a) (cfc g a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    ⊢ Commute (cfc f a) (cfc g a)
  -/
  refine cfc_cases (fun x ↦ Commute x (cfc g a)) a f (by simp) fun hf ha ↦ ?_
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : ContinuousOn f (spectrum R a)
    ha : p a
    ⊢ (fun x => Commute x (cfc g a)) ((cfcHom ha) { toFun := (spectrum R a).restri …
  -/
  refine cfc_cases (fun x ↦ Commute _ x) a g (by simp) fun hg _ ↦ ?_
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : ContinuousOn f (spectrum R a)
    ha : p a
    hg : ContinuousOn g (spectrum R a)
    x✝ : p a
    ⊢ (fun x => Commute ((cfcHom ha) { toFun := (spectrum R a).restrict f, continu …
  -/
  exact Commute.all _ _ |>.map _
  /-
    🎉 no goals
  -/


variable (R) in
lemma cfc_id (ha : p a := by cfc_tac) : cfc (id : R → R) a = a :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  /-
    🎉 no goals
  -/
  cfc_apply (id : R → R) a ▸ cfcHom_id (p := p) ha
  /-
    🎉 no goals
  -/


variable (R) in
                                                                      /-
                                                                        R : Type u_1
                                                                        A : Type u_2
                                                                        p : A → Prop
                                                                        inst✝⁸ : CommSemiring R
                                                                        inst✝⁷ : StarRing R
                                                                        inst✝⁶ : MetricSpace R
                                                                        inst✝⁵ : TopologicalSemiring R
                                                                        inst✝⁴ : ContinuousStar R
                                                                        inst✝³ : TopologicalSpace A
                                                                        inst✝² : Ring A
                                                                        inst✝¹ : StarRing A
                                                                        inst✝ : Algebra R A
                                                                        instCFC : ContinuousFunctionalCalculus R p
                                                                        a : A
                                                                        ha : autoParam (p a) _auto✝
                                                                        ⊢ p a
                                                                      -/
lemma cfc_id' (ha : p a := by cfc_tac) : cfc (fun x : R ↦ x) a = a := cfc_id R a
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The **spectral mapping theorem** for the continuous functional calculus. -/
lemma cfc_map_spectrum (ha : p a := by cfc_tac)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    spectrum R (cfc f a) = f '' spectrum R a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Eq (spectrum R (cfc f a)) (Set.image f (spectrum R a))
  -/
  simp [cfc_apply f a, cfcHom_map_spectrum (p := p)]
  /-
    🎉 no goals
  -/


lemma cfc_const (r : R) (a : A) (ha : p a := by cfc_tac) :
    cfc (fun _ ↦ r) a = algebraMap R A r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => r) a) ((algebraMap R A) r)
  -/
  rw [cfc_apply (fun _ : R ↦ r) a, ← AlgHomClass.commutes (cfcHom ha (p := p)) r]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => r, continuous_to …
  -/
  congr
  /-
    🎉 no goals
  -/


variable (R) in
include R in
lemma cfc_predicate_zero : p 0 :=
  ContinuousFunctionalCalculus.predicate_zero (R := R)


lemma cfc_predicate (f : R → R) (a : A) : p (cfc f a) :=
  cfc_cases p a f (cfc_predicate_zero R) fun _ _ ↦ cfcHom_predicate ..


lemma cfc_predicate_algebraMap (r : R) : p (algebraMap R A r) :=
  cfc_const r (0 : A) (cfc_predicate_zero R) ▸ cfc_predicate (fun _ ↦ r) 0


variable (R) in
include R in
lemma cfc_predicate_one : p 1 :=
  map_one (algebraMap R A) ▸ cfc_predicate_algebraMap (1 : R)


lemma cfc_congr {f g : R → R} {a : A} (hfg : (spectrum R a).EqOn f g) :
    cfc f a = cfc g a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hfg : Set.EqOn f g (spectrum R a)
    ⊢ Eq (cfc f a) (cfc g a)
  -/
  by_cases h : p a ∧ ContinuousOn g (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hfg : Set.EqOn f g (spectrum R a)
      h : And (p a) (ContinuousOn g (spectrum R a))
      ⊢ Eq (cfc f a) (cfc g a)
    -/
  · rw [cfc_apply (ha := h.1) (hf := h.2.congr hfg), cfc_apply (ha := h.1) (hf := h.2)]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hfg : Set.EqOn f g (spectrum R a)
      h : And (p a) (ContinuousOn g (spectrum R a))
      ⊢ Eq ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, continuous_toFun := ⋯ } …
    -/
    congr
    /-
      case pos.h.e_6.h.e_toFun
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hfg : Set.EqOn f g (spectrum R a)
      h : And (p a) (ContinuousOn g (spectrum R a))
      ⊢ Eq ((spectrum R a).restrict f) ((spectrum R a).restrict g)
    -/
    exact Set.restrict_eq_iff.mpr hfg
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hfg : Set.EqOn f g (spectrum R a)
      h : Not (And (p a) (ContinuousOn g (spectrum R a)))
      ⊢ Eq (cfc f a) (cfc g a)
    -/
  · obtain (ha | hg) := not_and_or.mp h
      /-
        case neg.inl
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f g : R → R
        a : A
        hfg : Set.EqOn f g (spectrum R a)
        h : Not (And (p a) (ContinuousOn g (spectrum R a)))
        ha : Not (p a)
        ⊢ Eq (cfc f a) (cfc g a)
      -/
    · simp [cfc_apply_of_not_predicate a ha]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f g : R → R
        a : A
        hfg : Set.EqOn f g (spectrum R a)
        h : Not (And (p a) (ContinuousOn g (spectrum R a)))
        hg : Not (ContinuousOn g (spectrum R a))
        ⊢ Eq (cfc f a) (cfc g a)
      -/
    · rw [cfc_apply_of_not_continuousOn a hg, cfc_apply_of_not_continuousOn]
      /-
        case neg.inr.hf
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f g : R → R
        a : A
        hfg : Set.EqOn f g (spectrum R a)
        h : Not (And (p a) (ContinuousOn g (spectrum R a)))
        hg : Not (ContinuousOn g (spectrum R a))
        ⊢ Not (ContinuousOn f (spectrum R a))
      -/
      exact fun hf ↦ hg (hf.congr hfg.symm)
      /-
        🎉 no goals
      -/


lemma eqOn_of_cfc_eq_cfc {f g : R → R} {a : A} (h : cfc f a = cfc g a)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    (spectrum R a).EqOn f g := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    h : Eq (cfc f a) (cfc g a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Set.EqOn f g (spectrum R a)
  -/
  rw [cfc_apply f a, cfc_apply g a] at h
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    h : Eq ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun :=  …
    ⊢ Set.EqOn f g (spectrum R a)
  -/
  have := (cfcHom_isClosedEmbedding (show p a from ha) (R := R)).injective h
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    h : Eq ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun :=  …
    this : Eq { toFun := (spectrum R a).restrict f, continuous_toFun := ⋯ } { toFu …
    ⊢ Set.EqOn f g (spectrum R a)
  -/
  intro x hx
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    h : Eq ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun :=  …
    this : Eq { toFun := (spectrum R a).restrict f, continuous_toFun := ⋯ } { toFu …
    x : R
    hx : Membership.mem (spectrum R a) x
    ⊢ Eq (f x) (g x)
  -/
  congrm($(this) ⟨x, hx⟩)
  /-
    🎉 no goals
  -/


variable {a f g} in
include ha hf hg in
lemma cfc_eq_cfc_iff_eqOn : cfc f a = cfc g a ↔ (spectrum R a).EqOn f g :=
   /-
     R : Type u_1
     A : Type u_2
     p : A → Prop
     inst✝⁸ : CommSemiring R
     inst✝⁷ : StarRing R
     inst✝⁶ : MetricSpace R
     inst✝⁵ : TopologicalSemiring R
     inst✝⁴ : ContinuousStar R
     inst✝³ : TopologicalSpace A
     inst✝² : Ring A
     inst✝¹ : StarRing A
     inst✝ : Algebra R A
     instCFC : ContinuousFunctionalCalculus R p
     f g : R → R
     a : A
     ha : autoParam (p a) _auto✝
     hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
     hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
     h : Eq (cfc f a) (cfc g a)
     ⊢ ContinuousOn f (spectrum R a)
   -/
   /-
     🎉 no goals
   -/
   /-
     🎉 no goals
   -/
  ⟨eqOn_of_cfc_eq_cfc, cfc_congr⟩
   /-
     🎉 no goals
   -/


variable (R)


include ha in
lemma cfc_one : cfc (1 : R → R) a = 1 :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  /-
    🎉 no goals
  -/
  cfc_apply (1 : R → R) a ▸ map_one (cfcHom (show p a from ha))
  /-
    🎉 no goals
  -/


include ha in
                                                   /-
                                                     R : Type u_1
                                                     A : Type u_2
                                                     p : A → Prop
                                                     inst✝⁸ : CommSemiring R
                                                     inst✝⁷ : StarRing R
                                                     inst✝⁶ : MetricSpace R
                                                     inst✝⁵ : TopologicalSemiring R
                                                     inst✝⁴ : ContinuousStar R
                                                     inst✝³ : TopologicalSpace A
                                                     inst✝² : Ring A
                                                     inst✝¹ : StarRing A
                                                     inst✝ : Algebra R A
                                                     instCFC : ContinuousFunctionalCalculus R p
                                                     a : A
                                                     ha : autoParam (p a) _auto✝
                                                     ⊢ p a
                                                   -/
lemma cfc_const_one : cfc (fun _ : R ↦ 1) a = 1 := cfc_one R a
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
lemma cfc_zero : cfc (0 : R → R) a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ⊢ Eq (cfc 0 a) 0
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : p a
      ⊢ Eq (cfc 0 a) 0
    -/
  · exact cfc_apply (0 : R → R) a ▸ map_zero (cfcHom ha)
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      ha : Not (p a)
      ⊢ Eq (cfc 0 a) 0
    -/
  · rw [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


@[simp]
lemma cfc_const_zero : cfc (fun _ : R ↦ 0) a = 0 :=
  cfc_zero R a


variable {R}


lemma cfc_mul (f g : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) :
    cfc (fun x ↦ f x * g x) a = cfc f a * cfc g a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ⊢ Eq (cfc (fun x => HMul.hMul (f x) (g x)) a) (HMul.hMul (cfc f a) (cfc g a))
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq (cfc (fun x => HMul.hMul (f x) (g x)) a) (HMul.hMul (cfc f a) (cfc g a))
    -/
  · rw [cfc_apply f a, cfc_apply g a, ← map_mul, cfc_apply _ a]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => HMul.hMul (f x)  …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : Not (p a)
      ⊢ Eq (cfc (fun x => HMul.hMul (f x) (g x)) a) (HMul.hMul (cfc f a) (cfc g a))
    -/
  · simp [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


lemma cfc_pow (f : R → R) (n : ℕ) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (fun x ↦ (f x) ^ n) a = cfc f a ^ n := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    n : Nat
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow (f x) n) a) (HPow.hPow (cfc f a) n)
  -/
  rw [cfc_apply f a, ← map_pow, cfc_apply _ a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    n : Nat
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => HPow.hPow (f x)  …
  -/
  congr
  /-
    🎉 no goals
  -/


lemma cfc_add (f g : R → R) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) :
    cfc (fun x ↦ f x + g x) a = cfc f a + cfc g a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    f g : R → R
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ⊢ Eq (cfc (fun x => HAdd.hAdd (f x) (g x)) a) (HAdd.hAdd (cfc f a) (cfc g a))
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      f g : R → R
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq (cfc (fun x => HAdd.hAdd (f x) (g x)) a) (HAdd.hAdd (cfc f a) (cfc g a))
    -/
  · rw [cfc_apply f a, cfc_apply g a, ← map_add, cfc_apply _ a]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      f g : R → R
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => HAdd.hAdd (f x)  …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      a : A
      f g : R → R
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : Not (p a)
      ⊢ Eq (cfc (fun x => HAdd.hAdd (f x) (g x)) a) (HAdd.hAdd (cfc f a) (cfc g a))
    -/
  · simp [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


lemma cfc_const_add (r : R) (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (fun x => r + f x) a = algebraMap R A r + cfc f a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HAdd.hAdd r (f x)) a) (HAdd.hAdd ((algebraMap R A) r) (cfc …
  -/
  have : (fun z => r + f z) = (fun z => (fun _ => r) z + f z) := by ext; simp
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    this : Eq (fun z => HAdd.hAdd r (f z)) fun z => HAdd.hAdd ((fun x => r) z) (f z)
    ⊢ Eq (cfc (fun x => HAdd.hAdd r (f x)) a) (HAdd.hAdd ((algebraMap R A) r) (cfc …
  -/
  rw [this, cfc_add a _ _ (continuousOn_const (c := r)) hf, cfc_const r a ha]
  /-
    🎉 no goals
  -/


lemma cfc_add_const (r : R) (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (fun x => f x + r) a = cfc f a + algebraMap R A r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HAdd.hAdd (f x) r) a) (HAdd.hAdd (cfc f a) ((algebraMap R  …
  -/
  rw [add_comm (cfc f a)]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HAdd.hAdd (f x) r) a) (HAdd.hAdd ((algebraMap R A) r) (cfc …
  -/
  conv_lhs => simp only [add_comm]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HAdd.hAdd r (f x)) a) (HAdd.hAdd ((algebraMap R A) r) (cfc …
  -/
  exact cfc_const_add r f a hf ha
  /-
    🎉 no goals
  -/


open Finset in
lemma cfc_sum {ι : Type*} (f : ι → R → R) (a : A) (s : Finset ι)
    (hf : ∀ i ∈ s, ContinuousOn (f i) (spectrum R a) := by cfc_cont_tac) :
    cfc (∑ i in s, f i)  a = ∑ i in s, cfc (f i) a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    ι : Type u_3
    f : ι → R → R
    a : A
    s : Finset ι
    hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
    ⊢ Eq (cfc (s.sum fun i => f i) a) (s.sum fun i => cfc (f i) a)
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      ⊢ Eq (cfc (s.sum fun i => f i) a) (s.sum fun i => cfc (f i) a)
    -/
  · have hsum : s.sum f = fun z => ∑ i ∈ s, f i z := by ext; simp
    have hf' : ContinuousOn (∑ i : s, f i) (spectrum R a) := by
      rw [sum_coe_sort s, hsum]
      exact continuousOn_finset_sum s fun i hi => hf i hi
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      hsum : Eq (s.sum f) fun z => s.sum fun i => f i z
      hf' : ContinuousOn (Finset.univ.sum fun i => f ↑i) (spectrum R a)
      ⊢ Eq (cfc (s.sum fun i => f i) a) (s.sum fun i => cfc (f i) a)
    -/
    rw [← sum_coe_sort s, ← sum_coe_sort s]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      hsum : Eq (s.sum f) fun z => s.sum fun i => f i z
      hf' : ContinuousOn (Finset.univ.sum fun i => f ↑i) (spectrum R a)
      ⊢ Eq (cfc (Finset.univ.sum fun i => f ↑i) a) (Finset.univ.sum fun i => cfc (f  …
    -/
    rw [cfc_apply_pi _ a _ (fun ⟨i, hi⟩ => hf i hi), ← map_sum, cfc_apply _ a ha hf']
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      hsum : Eq (s.sum f) fun z => s.sum fun i => f i z
      hf' : ContinuousOn (Finset.univ.sum fun i => f ↑i) (spectrum R a)
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict (Finset.univ.sum fun i => …
    -/
    congr 1
    /-
      case pos.h.e_6.h
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      hsum : Eq (s.sum f) fun z => s.sum fun i => f i z
      hf' : ContinuousOn (Finset.univ.sum fun i => f ↑i) (spectrum R a)
      ⊢ Eq { toFun := (spectrum R a).restrict (Finset.univ.sum fun i => f ↑i), conti …
    -/
    ext
    /-
      case pos.h.e_6.h.h
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : p a
      hsum : Eq (s.sum f) fun z => s.sum fun i => f i z
      hf' : ContinuousOn (Finset.univ.sum fun i => f ↑i) (spectrum R a)
      a✝ : ↑(spectrum R a)
      ⊢ Eq ({ toFun := (spectrum R a).restrict (Finset.univ.sum fun i => f ↑i), cont …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      ι : Type u_3
      f : ι → R → R
      a : A
      s : Finset ι
      hf : autoParam (∀ (i : ι), Membership.mem s i → ContinuousOn (f i) (spectrum R …
      ha : Not (p a)
      ⊢ Eq (cfc (s.sum fun i => f i) a) (s.sum fun i => cfc (f i) a)
    -/
  · simp [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


open Finset in
lemma cfc_sum_univ {ι : Type*} [Fintype ι] (f : ι → R → R) (a : A)
    (hf : ∀ i, ContinuousOn (f i) (spectrum R a) := by cfc_cont_tac) :
    cfc (∑ i, f i) a = ∑ i, cfc (f i) a :=
  cfc_sum f a _ fun i _ ↦ hf i


lemma cfc_smul {S : Type*} [SMul S R] [ContinuousConstSMul S R]
    [SMulZeroClass S A] [IsScalarTower S R A] [IsScalarTower S R (R → R)]
    (s : S) (f : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    cfc (fun x ↦ s • f x) a = s • cfc f a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : MetricSpace R
    inst✝¹⁰ : TopologicalSemiring R
    inst✝⁹ : ContinuousStar R
    inst✝⁸ : TopologicalSpace A
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    S : Type u_3
    inst✝⁴ : SMul S R
    inst✝³ : ContinuousConstSMul S R
    inst✝² : SMulZeroClass S A
    inst✝¹ : IsScalarTower S R A
    inst✝ : IsScalarTower S R (R → R)
    s : S
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Eq (cfc (fun x => HSMul.hSMul s (f x)) a) (HSMul.hSMul s (cfc f a))
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹³ : CommSemiring R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : Ring A
      inst✝⁶ : StarRing A
      inst✝⁵ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      S : Type u_3
      inst✝⁴ : SMul S R
      inst✝³ : ContinuousConstSMul S R
      inst✝² : SMulZeroClass S A
      inst✝¹ : IsScalarTower S R A
      inst✝ : IsScalarTower S R (R → R)
      s : S
      f : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq (cfc (fun x => HSMul.hSMul s (f x)) a) (HSMul.hSMul s (cfc f a))
    -/
  · rw [cfc_apply f a, cfc_apply _ a]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹³ : CommSemiring R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : Ring A
      inst✝⁶ : StarRing A
      inst✝⁵ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      S : Type u_3
      inst✝⁴ : SMul S R
      inst✝³ : ContinuousConstSMul S R
      inst✝² : SMulZeroClass S A
      inst✝¹ : IsScalarTower S R A
      inst✝ : IsScalarTower S R (R → R)
      s : S
      f : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => HSMul.hSMul s (f …
    -/
    simp_rw [← Pi.smul_def, ← smul_one_smul R s _]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹³ : CommSemiring R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : Ring A
      inst✝⁶ : StarRing A
      inst✝⁵ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      S : Type u_3
      inst✝⁴ : SMul S R
      inst✝³ : ContinuousConstSMul S R
      inst✝² : SMulZeroClass S A
      inst✝¹ : IsScalarTower S R A
      inst✝ : IsScalarTower S R (R → R)
      s : S
      f : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict (HSMul.hSMul (HSMul.hSMul …
    -/
    rw [← map_smul]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹³ : CommSemiring R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : Ring A
      inst✝⁶ : StarRing A
      inst✝⁵ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      S : Type u_3
      inst✝⁴ : SMul S R
      inst✝³ : ContinuousConstSMul S R
      inst✝² : SMulZeroClass S A
      inst✝¹ : IsScalarTower S R A
      inst✝ : IsScalarTower S R (R → R)
      s : S
      f : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict (HSMul.hSMul (HSMul.hSMul …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹³ : CommSemiring R
      inst✝¹² : StarRing R
      inst✝¹¹ : MetricSpace R
      inst✝¹⁰ : TopologicalSemiring R
      inst✝⁹ : ContinuousStar R
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : Ring A
      inst✝⁶ : StarRing A
      inst✝⁵ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      S : Type u_3
      inst✝⁴ : SMul S R
      inst✝³ : ContinuousConstSMul S R
      inst✝² : SMulZeroClass S A
      inst✝¹ : IsScalarTower S R A
      inst✝ : IsScalarTower S R (R → R)
      s : S
      f : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : Not (p a)
      ⊢ Eq (cfc (fun x => HSMul.hSMul s (f x)) a) (HSMul.hSMul s (cfc f a))
    -/
  · simp [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


lemma cfc_const_mul (r : R) (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    cfc (fun x ↦ r * f x) a = r • cfc f a :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ ContinuousOn f (spectrum R a)
  -/
  cfc_smul r f a
  /-
    🎉 no goals
  -/


lemma cfc_star (f : R → R) (a : A) : cfc (fun x ↦ star (f x)) a = star (cfc f a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
  -/
  by_cases h : p a ∧ ContinuousOn f (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : And (p a) (ContinuousOn f (spectrum R a))
      ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
    -/
  · obtain ⟨ha, hf⟩ := h
    /-
      case pos.intro
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      ha : p a
      hf : ContinuousOn f (spectrum R a)
      ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
    -/
    rw [cfc_apply f a, ← map_star, cfc_apply _ a]
    /-
      case pos.intro
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      ha : p a
      hf : ContinuousOn f (spectrum R a)
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => Star.star (f x), …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : Not (And (p a) (ContinuousOn f (spectrum R a)))
      ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
    -/
  · obtain (ha | hf) := not_and_or.mp h
      /-
        case neg.inl
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        ha : Not (p a)
        ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
      -/
    · simp [cfc_apply_of_not_predicate a ha]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        hf : Not (ContinuousOn f (spectrum R a))
        ⊢ Eq (cfc (fun x => Star.star (f x)) a) (Star.star (cfc f a))
      -/
    · rw [cfc_apply_of_not_continuousOn a hf, cfc_apply_of_not_continuousOn, star_zero]
      /-
        case neg.inr.hf
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        hf : Not (ContinuousOn f (spectrum R a))
        ⊢ Not (ContinuousOn (fun x => Star.star (f x)) (spectrum R a))
      -/
      exact fun hf_star ↦ hf <| by simpa using hf_star.star
      /-
        🎉 no goals
      -/


lemma cfc_pow_id (a : A) (n : ℕ) (ha : p a := by cfc_tac) : cfc (· ^ n : R → R) a = a ^ n := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    n : Nat
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HPow.hPow x n) a) (HPow.hPow a n)
  -/
  rw [cfc_pow .., cfc_id' ..]
  /-
    🎉 no goals
  -/


lemma cfc_smul_id {S : Type*} [SMul S R] [ContinuousConstSMul S R]
    [SMulZeroClass S A] [IsScalarTower S R A] [IsScalarTower S R (R → R)]
    (s : S) (a : A) (ha : p a := by cfc_tac) : cfc (s • · : R → R) a = s • a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : MetricSpace R
    inst✝¹⁰ : TopologicalSemiring R
    inst✝⁹ : ContinuousStar R
    inst✝⁸ : TopologicalSpace A
    inst✝⁷ : Ring A
    inst✝⁶ : StarRing A
    inst✝⁵ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    S : Type u_3
    inst✝⁴ : SMul S R
    inst✝³ : ContinuousConstSMul S R
    inst✝² : SMulZeroClass S A
    inst✝¹ : IsScalarTower S R A
    inst✝ : IsScalarTower S R (R → R)
    s : S
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HSMul.hSMul s x) a) (HSMul.hSMul s a)
  -/
  rw [cfc_smul .., cfc_id' ..]
  /-
    🎉 no goals
  -/


lemma cfc_const_mul_id (r : R) (a : A) (ha : p a := by cfc_tac) : cfc (r * ·) a = r • a :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  cfc_smul_id r a
  /-
    🎉 no goals
  -/


include ha in
lemma cfc_star_id : cfc (star · : R → R) a = star a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => Star.star x) a) (Star.star a)
  -/
  rw [cfc_star .., cfc_id' ..]
  /-
    🎉 no goals
  -/


lemma cfc_eval_X (ha : p a := by cfc_tac) : cfc (X : R[X]).eval a = a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun a => Polynomial.eval a Polynomial.X) a) a
  -/
  simpa using cfc_id R a
  /-
    🎉 no goals
  -/


lemma cfc_eval_C (r : R) (a : A) (ha : p a := by cfc_tac) :
    cfc (C r).eval a = algebraMap R A r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun a => Polynomial.eval a (Polynomial.C r)) a) ((algebraMap R A) r)
  -/
  simp [cfc_const r a]
  /-
    🎉 no goals
  -/


lemma cfc_map_polynomial (q : R[X]) (f : R → R) (a : A) (ha : p a := by cfc_tac)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    cfc (fun x ↦ q.eval (f x)) a = aeval (cfc f a) q := by
  induction q using Polynomial.induction_on with
  | h_C r => simp [cfc_const r a]
  | h_add q₁ q₂ hq₁ hq₂ =>
    simp only [eval_add, map_add, ← hq₁, ← hq₂, cfc_add a (q₁.eval <| f ·) (q₂.eval <| f ·)]
  | h_monomial n r _ =>
    simp only [eval_mul, eval_C, eval_pow, eval_X, map_mul, aeval_C, map_pow, aeval_X]
    rw [cfc_const_mul .., cfc_pow _ (n + 1) _, ← smul_eq_mul, algebraMap_smul]


lemma cfc_polynomial (q : R[X]) (a : A) (ha : p a := by cfc_tac) :
    cfc q.eval a = aeval a q := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    q : Polynomial R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun a => Polynomial.eval a q) a) ((Polynomial.aeval a) q)
  -/
  rw [cfc_map_polynomial .., cfc_id' ..]
  /-
    🎉 no goals
  -/


variable [UniqueContinuousFunctionalCalculus R A]


lemma cfc_comp (g f : R → R) (a : A) (ha : p a := by cfc_tac)
    (hg : ContinuousOn g (f '' spectrum R a) := by cfc_cont_tac)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    cfc (g ∘ f) a = cfc g (cfc f a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    g f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Eq (cfc (Function.comp g f) a) (cfc g (cfc f a))
  -/
  have := hg.comp hf <| (spectrum R a).mapsTo_image f
  have sp_eq : spectrum R (cfcHom (show p a from ha) (ContinuousMap.mk _ hf.restrict)) =
      f '' (spectrum R a) := by
    rw [cfcHom_map_spectrum (by exact ha) _]
    ext
    simp
  rw [cfc_apply .., cfc_apply f a,
    cfc_apply _ _ (cfcHom_predicate (show p a from ha) _) (by convert hg), ← cfcHom_comp _ _]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    g f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    this : ContinuousOn (Function.comp g f) (spectrum R a)
    sp_eq : Eq (spectrum R ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, conti …
    ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict (Function.comp g f), cont …
  -/
  swap
    /-
      case f'
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      g f : R → R
      a : A
      ha : autoParam (p a) _auto✝
      hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      this : ContinuousOn (Function.comp g f) (spectrum R a)
      sp_eq : Eq (spectrum R ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, conti …
      ⊢ ContinuousMap ↑(spectrum R a) ↑(spectrum R ((cfcHom ⋯) { toFun := (spectrum  …
    -/
  · exact ContinuousMap.mk _ <| hf.restrict.codRestrict fun x ↦ by rw [sp_eq]; use x.1; simp
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      g f : R → R
      a : A
      ha : autoParam (p a) _auto✝
      hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      this : ContinuousOn (Function.comp g f) (spectrum R a)
      sp_eq : Eq (spectrum R ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, conti …
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict (Function.comp g f), cont …
    -/
  · congr
    /-
      🎉 no goals
    -/
    /-
      case hff'
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalSemiring R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      inst✝ : UniqueContinuousFunctionalCalculus R A
      g f : R → R
      a : A
      ha : autoParam (p a) _auto✝
      hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      this : ContinuousOn (Function.comp g f) (spectrum R a)
      sp_eq : Eq (spectrum R ((cfcHom ⋯) { toFun := (spectrum R a).restrict f, conti …
      ⊢ ∀ (x : ↑(spectrum R a)), Eq ({ toFun := (spectrum R a).restrict f, continuou …
    -/
  · exact fun _ ↦ rfl
    /-
      🎉 no goals
    -/


lemma cfc_comp' (g f : R → R) (a : A) (hg : ContinuousOn g (f '' spectrum R a) := by cfc_cont_tac)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (g <| f ·) a = cfc g (cfc f a) :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    g f : R → R
    a : A
    hg : autoParam (ContinuousOn g (Set.image f (spectrum R a))) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  cfc_comp g f a
  /-
    🎉 no goals
  -/


lemma cfc_comp_pow (f : R → R) (n : ℕ) (a : A)
    (hf : ContinuousOn f ((· ^ n) '' (spectrum R a)) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (f <| · ^ n) a = cfc f (a ^ n) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : R → R
    n : Nat
    a : A
    hf : autoParam (ContinuousOn f (Set.image (fun x => HPow.hPow x n) (spectrum R …
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (HPow.hPow x n)) a) (cfc f (HPow.hPow a n))
  -/
  rw [cfc_comp' .., cfc_pow_id ..]
  /-
    🎉 no goals
  -/


lemma cfc_comp_smul {S : Type*} [SMul S R] [ContinuousConstSMul S R] [SMulZeroClass S A]
    [IsScalarTower S R A] [IsScalarTower S R (R → R)] (s : S) (f : R → R) (a : A)
    (hf : ContinuousOn f ((s • ·) '' (spectrum R a)) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (f <| s • ·) a = cfc f (s • a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁴ : CommSemiring R
    inst✝¹³ : StarRing R
    inst✝¹² : MetricSpace R
    inst✝¹¹ : TopologicalSemiring R
    inst✝¹⁰ : ContinuousStar R
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : Ring A
    inst✝⁷ : StarRing A
    inst✝⁶ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝⁵ : UniqueContinuousFunctionalCalculus R A
    S : Type u_3
    inst✝⁴ : SMul S R
    inst✝³ : ContinuousConstSMul S R
    inst✝² : SMulZeroClass S A
    inst✝¹ : IsScalarTower S R A
    inst✝ : IsScalarTower S R (R → R)
    s : S
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (Set.image (fun x => HSMul.hSMul s x) (spectrum …
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (HSMul.hSMul s x)) a) (cfc f (HSMul.hSMul s a))
  -/
  rw [cfc_comp' .., cfc_smul_id ..]
  /-
    🎉 no goals
  -/


lemma cfc_comp_const_mul (r : R) (f : R → R) (a : A)
    (hf : ContinuousOn f ((r * ·) '' (spectrum R a)) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (f <| r * ·) a = cfc f (r • a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    r : R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (Set.image (fun x => HMul.hMul r x) (spectrum R …
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (HMul.hMul r x)) a) (cfc f (HSMul.hSMul r a))
  -/
  rw [cfc_comp' .., cfc_const_mul_id ..]
  /-
    🎉 no goals
  -/


lemma cfc_comp_star (f : R → R) (a : A)
    (hf : ContinuousOn f (star '' (spectrum R a)) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (f <| star ·) a = cfc f (star a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (Set.image Star.star (spectrum R a))) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (Star.star x)) a) (cfc f (Star.star a))
  -/
  rw [cfc_comp' .., cfc_star_id ..]
  /-
    🎉 no goals
  -/


open Polynomial in
lemma cfc_comp_polynomial (q : R[X]) (f : R → R) (a : A)
    (hf : ContinuousOn f (q.eval '' (spectrum R a)) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (f <| q.eval ·) a = cfc f (aeval a q) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : UniqueContinuousFunctionalCalculus R A
    q : Polynomial R
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (Set.image (fun a => Polynomial.eval a q) (spec …
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (Polynomial.eval x q)) a) (cfc f ((Polynomial.aeval a) q))
  -/
  rw [cfc_comp' .., cfc_polynomial ..]
  /-
    🎉 no goals
  -/


lemma CFC.eq_algebraMap_of_spectrum_subset_singleton (r : R) (h_spec : spectrum R a ⊆ {r})
    (ha : p a := by cfc_tac) : a = algebraMap R A r := by
  simpa [cfc_id R a, cfc_const r a] using
    cfc_congr (f := id) (g := fun _ : R ↦ r) (a := a) fun x hx ↦ by simpa using h_spec hx


lemma CFC.eq_zero_of_spectrum_subset_zero (h_spec : spectrum R a ⊆ {0}) (ha : p a := by cfc_tac) :
    a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    h_spec : HasSubset.Subset (spectrum R a) (Singleton.singleton 0)
    ha : autoParam (p a) _auto✝
    ⊢ Eq a 0
  -/
  simpa using eq_algebraMap_of_spectrum_subset_singleton a 0 h_spec
  /-
    🎉 no goals
  -/


lemma CFC.eq_one_of_spectrum_subset_one (h_spec : spectrum R a ⊆ {1}) (ha : p a := by cfc_tac) :
    a = 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    h_spec : HasSubset.Subset (spectrum R a) (Singleton.singleton 1)
    ha : autoParam (p a) _auto✝
    ⊢ Eq a 1
  -/
  simpa using eq_algebraMap_of_spectrum_subset_singleton a 1 h_spec
  /-
    🎉 no goals
  -/


include instCFC in
lemma CFC.spectrum_algebraMap_subset (r : R) : spectrum R (algebraMap R A r) ⊆ {r} := by
  rw [← cfc_const r 0 (cfc_predicate_zero R),
    cfc_map_spectrum (fun _ ↦ r) 0 (cfc_predicate_zero R)]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    ⊢ HasSubset.Subset (Set.image (fun x => r) (spectrum R 0)) (Singleton.singleto …
  -/
  rintro - ⟨x, -, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r x : R
    ⊢ Membership.mem (Singleton.singleton r) ((fun x => r) x)
  -/
  simp
  /-
    🎉 no goals
  -/


include instCFC in
lemma CFC.spectrum_algebraMap_eq [Nontrivial A] (r : R) :
    spectrum R (algebraMap R A r) = {r} := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    r : R
    ⊢ Eq (spectrum R ((algebraMap R A) r)) (Singleton.singleton r)
  -/
  have hp : p 0 := cfc_predicate_zero R
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    r : R
    hp : p 0
    ⊢ Eq (spectrum R ((algebraMap R A) r)) (Singleton.singleton r)
  -/
  rw [← cfc_const r 0 hp, cfc_map_spectrum (fun _ => r) 0 hp]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    r : R
    hp : p 0
    ⊢ Eq (Set.image (fun x => r) (spectrum R 0)) (Singleton.singleton r)
  -/
  exact Set.Nonempty.image_const (⟨0, spectrum.zero_mem (R := R) not_isUnit_zero⟩) _
  /-
    🎉 no goals
  -/


include instCFC in
lemma CFC.spectrum_zero_eq [Nontrivial A] :
    spectrum R (0 : A) = {0} := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    ⊢ Eq (spectrum R 0) (Singleton.singleton 0)
  -/
  have : (0 : A) = algebraMap R A 0 := Eq.symm (RingHom.map_zero (algebraMap R A))
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    this : Eq 0 ((algebraMap R A) 0)
    ⊢ Eq (spectrum R 0) (Singleton.singleton 0)
  -/
  rw [this, spectrum_algebraMap_eq]
  /-
    🎉 no goals
  -/


include instCFC in
lemma CFC.spectrum_one_eq [Nontrivial A] :
    spectrum R (1 : A) = {1} := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    ⊢ Eq (spectrum R 1) (Singleton.singleton 1)
  -/
  have : (1 : A) = algebraMap R A 1 := Eq.symm (RingHom.map_one (algebraMap R A))
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    this : Eq 1 ((algebraMap R A) 1)
    ⊢ Eq (spectrum R 1) (Singleton.singleton 1)
  -/
  rw [this, spectrum_algebraMap_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma cfc_algebraMap (r : R) (f : R → R) : cfc f (algebraMap R A r) = algebraMap R A (f r) := by
  have h₁ : ContinuousOn f (spectrum R (algebraMap R A r)) :=
  continuousOn_singleton _ _ |>.mono <| CFC.spectrum_algebraMap_subset r
  rw [cfc_apply f (algebraMap R A r) (cfc_predicate_algebraMap r),
    ← AlgHomClass.commutes (cfcHom (p := p) (cfc_predicate_algebraMap r)) (f r)]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    h₁ : ContinuousOn f (spectrum R ((algebraMap R A) r))
    ⊢ Eq ((cfcHom ⋯) { toFun := (spectrum R ((algebraMap R A) r)).restrict f, cont …
  -/
  congr
  /-
    case h.e_6.h.e_toFun
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    h₁ : ContinuousOn f (spectrum R ((algebraMap R A) r))
    ⊢ Eq ((spectrum R ((algebraMap R A) r)).restrict f) fun x => (algebraMap R R)  …
  -/
  ext ⟨x, hx⟩
  /-
    case h.e_6.h.e_toFun.h.mk
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    h₁ : ContinuousOn f (spectrum R ((algebraMap R A) r))
    x : R
    hx : Membership.mem (spectrum R ((algebraMap R A) r)) x
    ⊢ Eq ((spectrum R ((algebraMap R A) r)).restrict f ⟨x, hx⟩) ((algebraMap R R)  …
  -/
  apply CFC.spectrum_algebraMap_subset r at hx
  /-
    case h.e_6.h.e_toFun.h.mk
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    f : R → R
    h₁ : ContinuousOn f (spectrum R ((algebraMap R A) r))
    x : R
    hx✝ : Membership.mem (spectrum R ((algebraMap R A) r)) x
    hx : Membership.mem (Singleton.singleton r) x
    ⊢ Eq ((spectrum R ((algebraMap R A) r)).restrict f ⟨x, hx✝⟩) ((algebraMap R R) …
  -/
  simp_all
  /-
    🎉 no goals
  -/


@[simp] lemma cfc_apply_zero {f : R → R} : cfc f (0 : A) = algebraMap R A (f 0) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    ⊢ Eq (cfc f 0) ((algebraMap R A) (f 0))
  -/
  simpa using cfc_algebraMap (A := A) 0 f
  /-
    🎉 no goals
  -/


@[simp] lemma cfc_apply_one {f : R → R} : cfc f (1 : A) = algebraMap R A (f 1) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : MetricSpace R
    inst✝⁵ : TopologicalSemiring R
    inst✝⁴ : ContinuousStar R
    inst✝³ : TopologicalSpace A
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    ⊢ Eq (cfc f 1) ((algebraMap R A) (f 1))
  -/
  simpa using cfc_algebraMap (A := A) 1 f
  /-
    🎉 no goals
  -/


@[simp]
instance IsStarNormal.cfc_map (f : R → R) (a : A) : IsStarNormal (cfc f a) where
  star_comm_self := by
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f✝ g : R → R
      a✝ : A
      ha : autoParam (p a✝) _auto✝
      hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
      f : R → R
      a : A
      ⊢ Commute (Star.star (cfc f a)) (cfc f a)
    -/
    rw [Commute, SemiconjBy]
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁸ : CommSemiring R
      inst✝⁷ : StarRing R
      inst✝⁶ : MetricSpace R
      inst✝⁵ : TopologicalSemiring R
      inst✝⁴ : ContinuousStar R
      inst✝³ : TopologicalSpace A
      inst✝² : Ring A
      inst✝¹ : StarRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f✝ g : R → R
      a✝ : A
      ha : autoParam (p a✝) _auto✝
      hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
      f : R → R
      a : A
      ⊢ Eq (HMul.hMul (Star.star (cfc f a)) (cfc f a)) (HMul.hMul (cfc f a) (Star.st …
    -/
    by_cases h : ContinuousOn f (spectrum R a)
      /-
        case pos
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f✝ g : R → R
        a✝ : A
        ha : autoParam (p a✝) _auto✝
        hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
        hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
        f : R → R
        a : A
        h : ContinuousOn f (spectrum R a)
        ⊢ Eq (HMul.hMul (Star.star (cfc f a)) (cfc f a)) (HMul.hMul (cfc f a) (Star.st …
      -/
    · rw [← cfc_star, ← cfc_mul .., ← cfc_mul ..]
      /-
        case pos
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f✝ g : R → R
        a✝ : A
        ha : autoParam (p a✝) _auto✝
        hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
        hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
        f : R → R
        a : A
        h : ContinuousOn f (spectrum R a)
        ⊢ Eq (cfc (fun x => HMul.hMul (Star.star (f x)) (f x)) a) (cfc (fun x => HMul. …
      -/
      congr! 2
      /-
        case pos.h.e'_14.h
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f✝ g : R → R
        a✝ : A
        ha : autoParam (p a✝) _auto✝
        hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
        hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
        f : R → R
        a : A
        h : ContinuousOn f (spectrum R a)
        x✝ : R
        ⊢ Eq (HMul.hMul (Star.star (f x✝)) (f x✝)) (HMul.hMul (f x✝) (Star.star (f x✝)))
      -/
      exact mul_comm _ _
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁸ : CommSemiring R
        inst✝⁷ : StarRing R
        inst✝⁶ : MetricSpace R
        inst✝⁵ : TopologicalSemiring R
        inst✝⁴ : ContinuousStar R
        inst✝³ : TopologicalSpace A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra R A
        instCFC : ContinuousFunctionalCalculus R p
        f✝ g : R → R
        a✝ : A
        ha : autoParam (p a✝) _auto✝
        hf : autoParam (ContinuousOn f✝ (spectrum R a✝)) _auto✝
        hg : autoParam (ContinuousOn g (spectrum R a✝)) _auto✝
        f : R → R
        a : A
        h : Not (ContinuousOn f (spectrum R a))
        ⊢ Eq (HMul.hMul (Star.star (cfc f a)) (cfc f a)) (HMul.hMul (cfc f a) (Star.st …
      -/
    · simp [cfc_apply_of_not_continuousOn a h]
      /-
        🎉 no goals
      -/

-- The following two lemmas are just `cfc_predicate`, but specific enough for the `@[simp]` tag.

@[simp]
protected lemma IsSelfAdjoint.cfc [ContinuousFunctionalCalculus R (IsSelfAdjoint : A → Prop)]
    {f : R → R} {a : A} : IsSelfAdjoint (cfc f a) :=
  cfc_predicate _ _


@[simp]
lemma cfc_nonneg_of_predicate [PartialOrder A]
    [ContinuousFunctionalCalculus R (fun (a : A) => 0 ≤ a)] {f : R → R} {a : A} : 0 ≤ cfc f a :=
  cfc_predicate _ _


variable (R) in
/-- In an `R`-algebra with a continuous functional calculus, every element satisfying the predicate
has nonempty `R`-spectrum. -/
lemma CFC.spectrum_nonempty [Nontrivial A] (a : A) (ha : p a := by cfc_tac) :
    (spectrum R a).Nonempty := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ (spectrum R a).Nonempty
  -/
  by_contra! h
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    a : A
    ha : autoParam (p a) _auto✝
    h : Eq (spectrum R a) EmptyCollection.emptyCollection
    ⊢ False
  -/
  apply one_ne_zero (α := A)
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    a : A
    ha : autoParam (p a) _auto✝
    h : Eq (spectrum R a) EmptyCollection.emptyCollection
    ⊢ Eq 1 0
  -/
  rw [← cfc_one R a, ← cfc_zero R a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommSemiring R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : Nontrivial A
    a : A
    ha : autoParam (p a) _auto✝
    h : Eq (spectrum R a) EmptyCollection.emptyCollection
    ⊢ Eq (cfc 1 a) (cfc 0 a)
  -/
  exact cfc_congr fun x hx ↦ by simp_all
  /-
    🎉 no goals
  -/


lemma isUnit_cfc_iff (f : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : IsUnit (cfc f a) ↔ ∀ x ∈ spectrum R a, f x ≠ 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : Semifield R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (IsUnit (cfc f a)) (∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x …
  -/
  rw [← spectrum.zero_not_mem_iff R, cfc_map_spectrum ..]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : Semifield R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalSemiring R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (Not (Membership.mem (Set.image f (spectrum R a)) 0)) (∀ (x : R), Member …
  -/
  aesop
  /-
    🎉 no goals
  -/


alias ⟨_, isUnit_cfc⟩ := isUnit_cfc_iff


/-- Bundle `cfc f a` into a unit given a proof that `f` is nonzero on the spectrum of `a`. -/
@[simps]
noncomputable def cfcUnits (hf' : ∀ x ∈ spectrum R a, f x ≠ 0)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) : Aˣ where
  val := cfc f a
  inv := cfc (fun x ↦ (f x)⁻¹) a
  val_inv := by
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      f : R → R
      a : A
      hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : autoParam (p a) _auto✝
      ⊢ Eq (HMul.hMul (cfc f a) (cfc (fun x => Inv.inv (f x)) a)) 1
    -/
    rw [← cfc_mul .., ← cfc_one R a]
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      f : R → R
      a : A
      hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : autoParam (p a) _auto✝
      ⊢ Eq (cfc (fun x => HMul.hMul (f x) (Inv.inv (f x))) a) (cfc 1 a)
    -/
    exact cfc_congr fun _ _ ↦ by aesop
    /-
      🎉 no goals
    -/
  inv_val := by
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      f : R → R
      a : A
      hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : autoParam (p a) _auto✝
      ⊢ Eq (HMul.hMul (cfc (fun x => Inv.inv (f x)) a) (cfc f a)) 1
    -/
    rw [← cfc_mul .., ← cfc_one R a]
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      f : R → R
      a : A
      hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      ha : autoParam (p a) _auto✝
      ⊢ Eq (cfc (fun x => HMul.hMul (Inv.inv (f x)) (f x)) a) (cfc 1 a)
    -/
    exact cfc_congr fun _ _ ↦ by aesop
    /-
      🎉 no goals
    -/


lemma cfcUnits_pow (hf' : ∀ x ∈ spectrum R a, f x ≠ 0) (n : ℕ)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
     /-
       R : Type u_1
       A : Type u_2
       p : A → Prop
       inst✝¹⁰ : Semifield R
       inst✝⁹ : StarRing R
       inst✝⁸ : MetricSpace R
       inst✝⁷ : TopologicalSemiring R
       inst✝⁶ : ContinuousStar R
       inst✝⁵ : TopologicalSpace A
       inst✝⁴ : Ring A
       inst✝³ : StarRing A
       inst✝² : Algebra R A
       inst✝¹ : ContinuousFunctionalCalculus R p
       inst✝ : HasContinuousInv₀ R
       f : R → R
       a : A
       hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
       n : Nat
       hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
       ha : autoParam (p a) _auto✝
       ⊢ ContinuousOn f (spectrum R a)
     -/
     /-
       🎉 no goals
     -/
    (cfcUnits f a hf') ^ n =
     /-
       🎉 no goals
     -/
      /-
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝¹⁰ : Semifield R
        inst✝⁹ : StarRing R
        inst✝⁸ : MetricSpace R
        inst✝⁷ : TopologicalSemiring R
        inst✝⁶ : ContinuousStar R
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra R A
        inst✝¹ : ContinuousFunctionalCalculus R p
        inst✝ : HasContinuousInv₀ R
        f : R → R
        a : A
        hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
        n : Nat
        hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
        ha : autoParam (p a) _auto✝
        ⊢ p a
      -/
      cfcUnits _ _ (forall₂_imp (fun _ _ ↦ pow_ne_zero n) hf') (hf := hf.pow n) := by
      /-
        🎉 no goals
      -/
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    f : R → R
    a : A
    hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
    n : Nat
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (HPow.hPow (cfcUnits f a hf' hf ha) n) (cfcUnits (fun x => HPow.hPow (f x …
  -/
  ext
  cases n with
  | zero => simp [cfc_const_one R a]
  | succ n => simp [cfc_pow f _ a]


lemma cfc_inv (hf' : ∀ x ∈ spectrum R a, f x ≠ 0)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (fun x ↦ (f x) ⁻¹) a = Ring.inverse (cfc f a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    f : R → R
    a : A
    hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => Inv.inv (f x)) a) (Ring.inverse (cfc f a))
  -/
  rw [← val_inv_cfcUnits f a hf', ← val_cfcUnits f a hf', Ring.inverse_unit]
  /-
    🎉 no goals
  -/


lemma cfc_inv_id (a : Aˣ) (ha : p a := by cfc_tac) :
    cfc (fun x ↦ x⁻¹ : R → R) (a : A) = a⁻¹ := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    a : Units A
    ha : autoParam (p ↑a) _auto✝
    ⊢ Eq (cfc (fun x => Inv.inv x) ↑a) ↑(Inv.inv a)
  -/
  rw [← Ring.inverse_unit]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    a : Units A
    ha : autoParam (p ↑a) _auto✝
    ⊢ Eq (cfc (fun x => Inv.inv x) ↑a) (Ring.inverse ↑a)
  -/
  convert cfc_inv (id : R → R) (a : A) ?_
    /-
      case h.e'_3.h.e'_3
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      a : Units A
      ha : autoParam (p ↑a) _auto✝
      ⊢ Eq (↑a) (cfc id ↑a)
    -/
  · exact (cfc_id R (a : A)).symm
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      a : Units A
      ha : autoParam (p ↑a) _auto✝
      ⊢ ∀ (x : R), Membership.mem (spectrum R ↑a) x → Ne (id x) 0
    -/
  · rintro x hx rfl
    /-
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹⁰ : Semifield R
      inst✝⁹ : StarRing R
      inst✝⁸ : MetricSpace R
      inst✝⁷ : TopologicalSemiring R
      inst✝⁶ : ContinuousStar R
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra R A
      inst✝¹ : ContinuousFunctionalCalculus R p
      inst✝ : HasContinuousInv₀ R
      a : Units A
      ha : autoParam (p ↑a) _auto✝
      hx : Membership.mem (spectrum R ↑a) 0
      ⊢ False
    -/
    exact spectrum.zero_not_mem R a.isUnit hx
    /-
      🎉 no goals
    -/


lemma cfc_map_div (f g : R → R) (a : A) (hg' : ∀ x ∈ spectrum R a, g x ≠ 0)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc (fun x ↦ f x / g x) a = cfc f a * Ring.inverse (cfc g a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    f g : R → R
    a : A
    hg' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (g x) 0
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HDiv.hDiv (f x) (g x)) a) (HMul.hMul (cfc f a) (Ring.inver …
  -/
  simp only [div_eq_mul_inv]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : Semifield R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalSemiring R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    inst✝ : HasContinuousInv₀ R
    f g : R → R
    a : A
    hg' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (g x) 0
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => HMul.hMul (f x) (Inv.inv (g x))) a) (HMul.hMul (cfc f a) ( …
  -/
  rw [cfc_mul .., cfc_inv g a hg']
  /-
    🎉 no goals
  -/


@[fun_prop]
lemma Units.continuousOn_inv₀_spectrum (a : Aˣ) : ContinuousOn (· ⁻¹) (spectrum R (a : A)) :=
  continuousOn_inv₀.mono <| by
    /-
      R : Type u_3
      A : Type u_4
      inst✝⁴ : Semifield R
      inst✝³ : Ring A
      inst✝² : TopologicalSpace R
      inst✝¹ : HasContinuousInv₀ R
      inst✝ : Algebra R A
      a : Units A
      ⊢ HasSubset.Subset (spectrum R ↑a) (HasCompl.compl (Singleton.singleton 0))
    -/
    simpa only [Set.subset_compl_singleton_iff] using spectrum.zero_not_mem R a.isUnit
    /-
      🎉 no goals
    -/


@[fun_prop]
lemma Units.continuousOn_zpow₀_spectrum [ContinuousMul R] (a : Aˣ) (n : ℤ) :
    ContinuousOn (· ^ n) (spectrum R (a : A)) :=
  (continuousOn_zpow₀ n).mono <| by
    /-
      R : Type u_3
      A : Type u_4
      inst✝⁵ : Semifield R
      inst✝⁴ : Ring A
      inst✝³ : TopologicalSpace R
      inst✝² : HasContinuousInv₀ R
      inst✝¹ : Algebra R A
      inst✝ : ContinuousMul R
      a : Units A
      n : Int
      ⊢ HasSubset.Subset (spectrum R ↑a) (HasCompl.compl (Singleton.singleton 0))
    -/
    simpa only [Set.subset_compl_singleton_iff] using spectrum.zero_not_mem R a.isUnit
    /-
      🎉 no goals
    -/


lemma cfcUnits_zpow (hf' : ∀ x ∈ spectrum R a, f x ≠ 0) (n : ℤ)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
     /-
       R : Type u_1
       A : Type u_2
       p : A → Prop
       inst✝¹⁰ : Semifield R
       inst✝⁹ : StarRing R
       inst✝⁸ : MetricSpace R
       inst✝⁷ : TopologicalSemiring R
       inst✝⁶ : ContinuousStar R
       inst✝⁵ : TopologicalSpace A
       inst✝⁴ : Ring A
       inst✝³ : StarRing A
       inst✝² : Algebra R A
       inst✝¹ : ContinuousFunctionalCalculus R p
       inst✝ : HasContinuousInv₀ R
       f : R → R
       a : A
       hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
       n : Int
       hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
       ha : autoParam (p a) _auto✝
       ⊢ ContinuousOn f (spectrum R a)
     -/
     /-
       🎉 no goals
     -/
    (cfcUnits f a hf') ^ n =
     /-
       🎉 no goals
     -/
      /-
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝¹⁰ : Semifield R
        inst✝⁹ : StarRing R
        inst✝⁸ : MetricSpace R
        inst✝⁷ : TopologicalSemiring R
        inst✝⁶ : ContinuousStar R
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra R A
        inst✝¹ : ContinuousFunctionalCalculus R p
        inst✝ : HasContinuousInv₀ R
        f : R → R
        a : A
        hf' : ∀ (x : R), Membership.mem (spectrum R a) x → Ne (f x) 0
        n : Int
        hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
        ha : autoParam (p a) _auto✝
        ⊢ p a
      -/
      cfcUnits (f ^ n) a (forall₂_imp (fun _ _ ↦ zpow_ne_zero n) hf')
      /-
        🎉 no goals
      -/
        (hf.zpow₀ n (forall₂_imp (fun _ _ ↦ Or.inl) hf')) := by
  cases n with
  | ofNat _ => simpa using cfcUnits_pow f a hf' _
  | negSucc n =>
    simp only [zpow_negSucc, ← inv_pow]
    ext
    exact cfc_pow (hf := hf.inv₀ hf') .. |>.symm


lemma cfc_zpow (a : Aˣ) (n : ℤ) (ha : p a := by cfc_tac) :
    cfc (fun x : R ↦ x ^ n) (a : A) = ↑(a ^ n) := by
  cases n with
  | ofNat n => simpa using cfc_pow_id (a : A) n
  | negSucc n =>
    simp only [zpow_negSucc, ← inv_pow, Units.val_pow_eq_pow_val]
    have := cfc_pow (fun x ↦ x⁻¹ : R → R) (n + 1) (a : A)
    exact this.trans <| congr($(cfc_inv_id a) ^ (n + 1))


lemma cfc_comp_inv (f : R → R) (a : Aˣ)
    (hf : ContinuousOn f ((· ⁻¹) '' (spectrum R (a : A))) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) :
    cfc (fun x ↦ f x⁻¹) (a : A) = cfc f (↑a⁻¹ : A) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : Semifield R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : Algebra R A
    inst✝² : ContinuousFunctionalCalculus R p
    inst✝¹ : HasContinuousInv₀ R
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : R → R
    a : Units A
    hf : autoParam (ContinuousOn f (Set.image (fun x => Inv.inv x) (spectrum R ↑a) …
    ha : autoParam (p ↑a) _auto✝
    ⊢ Eq (cfc (fun x => f (Inv.inv x)) ↑a) (cfc f ↑(Inv.inv a))
  -/
  rw [cfc_comp' .., cfc_inv_id _]
  /-
    🎉 no goals
  -/


lemma cfc_comp_zpow (f : R → R) (n : ℤ) (a : Aˣ)
    (hf : ContinuousOn f ((· ^ n) '' (spectrum R (a : A))) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) :
    cfc (fun x ↦ f (x ^ n)) (a : A) = cfc f (↑(a ^ n) : A) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : Semifield R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : Algebra R A
    inst✝² : ContinuousFunctionalCalculus R p
    inst✝¹ : HasContinuousInv₀ R
    inst✝ : UniqueContinuousFunctionalCalculus R A
    f : R → R
    n : Int
    a : Units A
    hf : autoParam (ContinuousOn f (Set.image (fun x => HPow.hPow x n) (spectrum R …
    ha : autoParam (p ↑a) _auto✝
    ⊢ Eq (cfc (fun x => f (HPow.hPow x n)) ↑a) (cfc f ↑(HPow.hPow a n))
  -/
  rw [cfc_comp' .., cfc_zpow a]
  /-
    🎉 no goals
  -/


variable (f g : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)

include hf hg in
lemma cfc_sub : cfc (fun x ↦ f x - g x) a = cfc f a - cfc g a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommRing R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalRing R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ⊢ Eq (cfc (fun x => HSub.hSub (f x) (g x)) a) (HSub.hSub (cfc f a) (cfc g a))
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq (cfc (fun x => HSub.hSub (f x) (g x)) a) (HSub.hSub (cfc f a) (cfc g a))
    -/
  · rw [cfc_apply f a, cfc_apply g a, ← map_sub, cfc_apply ..]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => HSub.hSub (f x)  …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : Not (p a)
      ⊢ Eq (cfc (fun x => HSub.hSub (f x) (g x)) a) (HSub.hSub (cfc f a) (cfc g a))
    -/
  · simp [cfc_apply_of_not_predicate a ha]
    /-
      🎉 no goals
    -/


lemma cfc_neg : cfc (fun x ↦ - (f x)) a = - (cfc f a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommRing R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalRing R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
  -/
  by_cases h : p a ∧ ContinuousOn f (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : And (p a) (ContinuousOn f (spectrum R a))
      ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
    -/
  · obtain ⟨ha, hf⟩ := h
    /-
      case pos.intro
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      ha : p a
      hf : ContinuousOn f (spectrum R a)
      ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
    -/
    rw [cfc_apply f a, ← map_neg, cfc_apply ..]
    /-
      case pos.intro
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      ha : p a
      hf : ContinuousOn f (spectrum R a)
      ⊢ Eq ((cfcHom ha) { toFun := (spectrum R a).restrict fun x => Neg.neg (f x), c …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝⁹ : CommRing R
      inst✝⁸ : StarRing R
      inst✝⁷ : MetricSpace R
      inst✝⁶ : TopologicalRing R
      inst✝⁵ : ContinuousStar R
      inst✝⁴ : TopologicalSpace A
      inst✝³ : Ring A
      inst✝² : StarRing A
      inst✝¹ : Algebra R A
      inst✝ : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : Not (And (p a) (ContinuousOn f (spectrum R a)))
      ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
    -/
  · obtain (ha | hf) := not_and_or.mp h
      /-
        case neg.inl
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁹ : CommRing R
        inst✝⁸ : StarRing R
        inst✝⁷ : MetricSpace R
        inst✝⁶ : TopologicalRing R
        inst✝⁵ : ContinuousStar R
        inst✝⁴ : TopologicalSpace A
        inst✝³ : Ring A
        inst✝² : StarRing A
        inst✝¹ : Algebra R A
        inst✝ : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        ha : Not (p a)
        ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
      -/
    · simp [cfc_apply_of_not_predicate a ha]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁹ : CommRing R
        inst✝⁸ : StarRing R
        inst✝⁷ : MetricSpace R
        inst✝⁶ : TopologicalRing R
        inst✝⁵ : ContinuousStar R
        inst✝⁴ : TopologicalSpace A
        inst✝³ : Ring A
        inst✝² : StarRing A
        inst✝¹ : Algebra R A
        inst✝ : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        hf : Not (ContinuousOn f (spectrum R a))
        ⊢ Eq (cfc (fun x => Neg.neg (f x)) a) (Neg.neg (cfc f a))
      -/
    · rw [cfc_apply_of_not_continuousOn a hf, cfc_apply_of_not_continuousOn, neg_zero]
      /-
        case neg.inr.hf
        R : Type u_1
        A : Type u_2
        p : A → Prop
        inst✝⁹ : CommRing R
        inst✝⁸ : StarRing R
        inst✝⁷ : MetricSpace R
        inst✝⁶ : TopologicalRing R
        inst✝⁵ : ContinuousStar R
        inst✝⁴ : TopologicalSpace A
        inst✝³ : Ring A
        inst✝² : StarRing A
        inst✝¹ : Algebra R A
        inst✝ : ContinuousFunctionalCalculus R p
        f : R → R
        a : A
        h : Not (And (p a) (ContinuousOn f (spectrum R a)))
        hf : Not (ContinuousOn f (spectrum R a))
        ⊢ Not (ContinuousOn (fun x => Neg.neg (f x)) (spectrum R a))
      -/
      exact fun hf_neg ↦ hf <| by simpa using hf_neg.neg
      /-
        🎉 no goals
      -/


lemma cfc_neg_id (ha : p a := by cfc_tac) : cfc (- · : R → R) a = -a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommRing R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalRing R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => Neg.neg x) a) (Neg.neg a)
  -/
  rw [cfc_neg _ a, cfc_id' R a]
  /-
    🎉 no goals
  -/


lemma cfc_comp_neg (hf : ContinuousOn f ((- ·) '' (spectrum R (a : A))) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : cfc (f <| - ·) a = cfc f (-a) := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹⁰ : CommRing R
    inst✝⁹ : StarRing R
    inst✝⁸ : MetricSpace R
    inst✝⁷ : TopologicalRing R
    inst✝⁶ : ContinuousStar R
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra R A
    inst✝¹ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    inst✝ : UniqueContinuousFunctionalCalculus R A
    hf : autoParam (ContinuousOn f (Set.image (fun x => Neg.neg x) (spectrum R a)) …
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (fun x => f (Neg.neg x)) a) (cfc f (Neg.neg a))
  -/
  rw [cfc_comp' .., cfc_neg_id _]
  /-
    🎉 no goals
  -/


lemma cfcHom_mono {a : A} (ha : p a) {f g : C(spectrum R a, R)} (hfg : f ≤ g) :
    cfcHom ha f ≤ cfcHom ha g :=
  OrderHomClass.mono (cfcHom ha) hfg


lemma cfcHom_nonneg_iff [NonnegSpectrumClass R A] {a : A} (ha : p a) {f : C(spectrum R a, R)} :
    0 ≤ cfcHom ha f ↔ 0 ≤ f := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalSemiring R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.632915) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : p a
    f : ContinuousMap (↑(spectrum R a)) R
    ⊢ Iff (LE.le 0 ((cfcHom ha) f)) (LE.le 0 f)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹² : OrderedCommSemiring R
      inst✝¹¹ : StarRing R
      inst✝¹⁰ : MetricSpace R
      inst✝⁹ : TopologicalSemiring R
      inst✝⁸ : ContinuousStar R
      inst✝⁷ : ∀ (α : Type ?u.632915) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : PartialOrder A
      inst✝² : StarOrderedRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      inst✝ : NonnegSpectrumClass R A
      a : A
      ha : p a
      f : ContinuousMap (↑(spectrum R a)) R
      ⊢ LE.le 0 ((cfcHom ha) f) → LE.le 0 f
    -/
  · exact fun hf x ↦ (cfcHom_map_spectrum ha (R := R) _ ▸ spectrum_nonneg_of_nonneg hf) ⟨x, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹² : OrderedCommSemiring R
      inst✝¹¹ : StarRing R
      inst✝¹⁰ : MetricSpace R
      inst✝⁹ : TopologicalSemiring R
      inst✝⁸ : ContinuousStar R
      inst✝⁷ : ∀ (α : Type ?u.632915) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : PartialOrder A
      inst✝² : StarOrderedRing A
      inst✝¹ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      inst✝ : NonnegSpectrumClass R A
      a : A
      ha : p a
      f : ContinuousMap (↑(spectrum R a)) R
      ⊢ LE.le 0 f → LE.le 0 ((cfcHom ha) f)
    -/
  · simpa using (cfcHom_mono ha (f := 0) (g := f) ·)
    /-
      🎉 no goals
    -/


lemma cfc_mono {f g : R → R} {a : A} (h : ∀ x ∈ spectrum R a, f x ≤ g x)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) :
    cfc f a ≤ cfc g a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.651788) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f g : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) (g x)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ⊢ LE.le (cfc f a) (cfc g a)
  -/
  by_cases ha : p a
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type ?u.651788) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) (g x)
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ LE.le (cfc f a) (cfc g a)
    -/
  · rw [cfc_apply f a, cfc_apply g a]
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type ?u.651788) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) (g x)
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : p a
      ⊢ LE.le ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun := …
    -/
    exact cfcHom_mono ha fun x ↦ h x.1 x.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f g : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) (g x)
      hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
      hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
      ha : Not (p a)
      ⊢ LE.le (cfc f a) (cfc g a)
    -/
  · simp only [cfc_apply_of_not_predicate _ ha, le_rfl]
    /-
      🎉 no goals
    -/


lemma cfc_nonneg_iff [NonnegSpectrumClass R A] (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : 0 ≤ cfc f a ↔ ∀ x ∈ spectrum R a, 0 ≤ f x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalSemiring R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.658127) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le 0 (cfc f a)) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le  …
  -/
  rw [cfc_apply .., cfcHom_nonneg_iff, ContinuousMap.le_def]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalSemiring R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (∀ (a_1 : ↑(spectrum R a)), LE.le (0 a_1) ({ toFun := (spectrum R a).res …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma StarOrderedRing.nonneg_iff_spectrum_nonneg [NonnegSpectrumClass R A] (a : A)
    (ha : p a := by cfc_tac) : 0 ≤ a ↔ ∀ x ∈ spectrum R a, 0 ≤ x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalSemiring R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.679294) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le 0 a) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le 0 x)
  -/
  have := cfc_nonneg_iff (id : R → R) a (by fun_prop) ha
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalSemiring R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : autoParam (p a) _auto✝
    this : Iff (LE.le 0 (cfc id a)) (∀ (x : R), Membership.mem (spectrum R a) x →  …
    ⊢ Iff (LE.le 0 a) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le 0 x)
  -/
  simpa [cfc_id _ a ha] using this
  /-
    🎉 no goals
  -/


lemma cfc_nonneg {f : R → R} {a : A} (h : ∀ x ∈ spectrum R a, 0 ≤ f x) :
    0 ≤ cfc f a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.692403) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le 0 (f x)
    ⊢ LE.le 0 (cfc f a)
  -/
  by_cases hf : ContinuousOn f (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type ?u.692403) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le 0 (f x)
      hf : ContinuousOn f (spectrum R a)
      ⊢ LE.le 0 (cfc f a)
    -/
  · simpa using cfc_mono h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le 0 (f x)
      hf : Not (ContinuousOn f (spectrum R a))
      ⊢ LE.le 0 (cfc f a)
    -/
  · simp only [cfc_apply_of_not_continuousOn _ hf, le_rfl]
    /-
      🎉 no goals
    -/


lemma cfc_nonpos (f : R → R) (a : A) (h : ∀ x ∈ spectrum R a, f x ≤ 0) :
    cfc f a ≤ 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.700653) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 0
    ⊢ LE.le (cfc f a) 0
  -/
  by_cases hf : ContinuousOn f (spectrum R a)
    /-
      case pos
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type ?u.700653) [inst : TopologicalSpace α], StarOrderedRing ( …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 0
      hf : ContinuousOn f (spectrum R a)
      ⊢ LE.le (cfc f a) 0
    -/
  · simpa using cfc_mono h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      A : Type u_2
      p : A → Prop
      inst✝¹¹ : OrderedCommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : MetricSpace R
      inst✝⁸ : TopologicalSemiring R
      inst✝⁷ : ContinuousStar R
      inst✝⁶ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : PartialOrder A
      inst✝¹ : StarOrderedRing A
      inst✝ : Algebra R A
      instCFC : ContinuousFunctionalCalculus R p
      f : R → R
      a : A
      h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 0
      hf : Not (ContinuousOn f (spectrum R a))
      ⊢ LE.le (cfc f a) 0
    -/
  · simp only [cfc_apply_of_not_continuousOn _ hf, le_rfl]
    /-
      🎉 no goals
    -/


lemma cfc_le_algebraMap (f : R → R) (r : R) (a : A) (h : ∀ x ∈ spectrum R a, f x ≤ r)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc f a ≤ algebraMap R A r :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.707480) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) r
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  /-
    🎉 no goals
  -/
                  /-
                    🎉 no goals
                  -/
  cfc_const r a ▸ cfc_mono h
                  /-
                    🎉 no goals
                  -/


lemma algebraMap_le_cfc (f : R → R) (r : R) (a : A) (h : ∀ x ∈ spectrum R a, r ≤ f x)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    algebraMap R A r ≤ cfc f a :=
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.712679) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le r (f x)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ p a
  -/
  /-
    🎉 no goals
  -/
                  /-
                    🎉 no goals
                  -/
  cfc_const r a ▸ cfc_mono h
                  /-
                    🎉 no goals
                  -/


lemma le_algebraMap_of_spectrum_le {r : R} {a : A} (h : ∀ x ∈ spectrum R a, x ≤ r)
    (ha : p a := by cfc_tac) : a ≤ algebraMap R A r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.717878) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le x r
    ha : autoParam (p a) _auto✝
    ⊢ LE.le a ((algebraMap R A) r)
  -/
  rw [← cfc_id R a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.717878) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le x r
    ha : autoParam (p a) _auto✝
    ⊢ LE.le (cfc id a) ((algebraMap R A) r)
  -/
  exact cfc_le_algebraMap id r a h
  /-
    🎉 no goals
  -/


lemma algebraMap_le_of_le_spectrum {r : R} {a : A} (h : ∀ x ∈ spectrum R a, r ≤ x)
    (ha : p a := by cfc_tac) : algebraMap R A r ≤ a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.722226) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le r x
    ha : autoParam (p a) _auto✝
    ⊢ LE.le ((algebraMap R A) r) a
  -/
  rw [← cfc_id R a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.722226) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    r : R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le r x
    ha : autoParam (p a) _auto✝
    ⊢ LE.le ((algebraMap R A) r) (cfc id a)
  -/
  exact algebraMap_le_cfc id r a h
  /-
    🎉 no goals
  -/


lemma cfc_le_one (f : R → R) (a : A) (h : ∀ x ∈ spectrum R a, f x ≤ 1) : cfc f a ≤ 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.726564) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 1
    ⊢ LE.le (cfc f a) 1
  -/
  apply cfc_cases (· ≤ 1) _ _ (by simp) fun hf ha ↦ ?_
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.726564) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 1
    hf : ContinuousOn f (spectrum R a)
    ha : p a
    ⊢ LE.le ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun := …
  -/
  rw [← map_one (cfcHom ha (R := R))]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.726564) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 1
    hf : ContinuousOn f (spectrum R a)
    ha : p a
    ⊢ LE.le ((cfcHom ha) { toFun := (spectrum R a).restrict f, continuous_toFun := …
  -/
  apply cfcHom_mono ha
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le (f x) 1
    hf : ContinuousOn f (spectrum R a)
    ha : p a
    ⊢ LE.le { toFun := (spectrum R a).restrict f, continuous_toFun := ⋯ } 1
  -/
  simpa [ContinuousMap.le_def] using h
  /-
    🎉 no goals
  -/


lemma one_le_cfc (f : R → R) (a : A) (h : ∀ x ∈ spectrum R a, 1 ≤ f x)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    1 ≤ cfc f a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.750116) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le 1 (f x)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ LE.le 1 (cfc f a)
  -/
  simpa using algebraMap_le_cfc f 1 a h
  /-
    🎉 no goals
  -/


lemma CFC.le_one {a : A} (h : ∀ x ∈ spectrum R a, x ≤ 1) (ha : p a := by cfc_tac) :
    a ≤ 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.757835) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le x 1
    ha : autoParam (p a) _auto✝
    ⊢ LE.le a 1
  -/
  simpa using le_algebraMap_of_spectrum_le h
  /-
    🎉 no goals
  -/


lemma CFC.one_le {a : A} (h : ∀ x ∈ spectrum R a, 1 ≤ x) (ha : p a := by cfc_tac) :
    1 ≤ a := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹¹ : OrderedCommSemiring R
    inst✝¹⁰ : StarRing R
    inst✝⁹ : MetricSpace R
    inst✝⁸ : TopologicalSemiring R
    inst✝⁷ : ContinuousStar R
    inst✝⁶ : ∀ (α : Type ?u.764937) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : StarOrderedRing A
    inst✝ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    a : A
    h : ∀ (x : R), Membership.mem (spectrum R a) x → LE.le 1 x
    ha : autoParam (p a) _auto✝
    ⊢ LE.le 1 a
  -/
  simpa using algebraMap_le_of_le_spectrum h
  /-
    🎉 no goals
  -/


lemma CFC.inv_nonneg_of_nonneg (a : Aˣ) (ha : (0 : A) ≤ a := by cfc_tac) : (0 : A) ≤ a⁻¹ :=
  /-
    A : Type u_1
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : PartialOrder A
    inst✝¹ : Algebra NNReal A
    inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
    a : Units A
    ha : autoParam (LE.le 0 ↑a) _auto✝
    ⊢ LE.le 0 ↑a
  -/
  cfc_inv_id (R := ℝ≥0) a ▸ cfc_predicate _ (a : A)
  /-
    🎉 no goals
  -/


lemma CFC.inv_nonneg (a : Aˣ)  : (0 : A) ≤ a⁻¹ ↔ (0 : A) ≤ a :=
                       /-
                         A : Type u_1
                         inst✝⁵ : TopologicalSpace A
                         inst✝⁴ : Ring A
                         inst✝³ : StarRing A
                         inst✝² : PartialOrder A
                         inst✝¹ : Algebra NNReal A
                         inst✝ : ContinuousFunctionalCalculus NNReal fun a => LE.le 0 a
                         a : Units A
                         x✝ : LE.le 0 ↑(Inv.inv a)
                         ⊢ LE.le 0 ↑(Inv.inv a)
                       -/
                       /-
                         🎉 no goals
                       -/
  ⟨fun _ ↦ inv_inv a ▸ inv_nonneg_of_nonneg a⁻¹, fun _ ↦ inv_nonneg_of_nonneg a⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma cfcHom_le_iff {a : A} (ha : p a) {f g : C(spectrum R a, R)} :
    cfcHom ha f ≤ cfcHom ha g ↔ f ≤ g := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.787138) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : p a
    f g : ContinuousMap (↑(spectrum R a)) R
    ⊢ Iff (LE.le ((cfcHom ha) f) ((cfcHom ha) g)) (LE.le f g)
  -/
  rw [← sub_nonneg, ← map_sub, cfcHom_nonneg_iff, sub_nonneg]
  /-
    🎉 no goals
  -/


lemma cfc_le_iff (f g : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hg : ContinuousOn g (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc f a ≤ cfc g a ↔ ∀ x ∈ spectrum R a, f x ≤ g x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.798779) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le (cfc f a) (cfc g a)) (∀ (x : R), Membership.mem (spectrum R a) x  …
  -/
  rw [cfc_apply f a, cfc_apply g a, cfcHom_le_iff (show p a from ha), ContinuousMap.le_def]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type u_1) [inst : TopologicalSpace α], StarOrderedRing (Contin …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f g : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hg : autoParam (ContinuousOn g (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (∀ (a_1 : ↑(spectrum R a)), LE.le ({ toFun := (spectrum R a).restrict f, …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma cfc_nonpos_iff (f : R → R) (a : A) (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : cfc f a ≤ 0 ↔ ∀ x ∈ spectrum R a, f x ≤ 0 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.815098) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le (cfc f a) 0) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le  …
  -/
  simp_rw [← neg_nonneg, ← cfc_neg]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.815098) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le 0 (cfc (fun x => Neg.neg (f x)) a)) (∀ (x : R), Membership.mem (s …
  -/
  exact cfc_nonneg_iff (fun x ↦ -f x) a
  /-
    🎉 no goals
  -/


lemma cfc_le_algebraMap_iff (f : R → R) (r : R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc f a ≤ algebraMap R A r ↔ ∀ x ∈ spectrum R a, f x ≤ r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.821834) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    r : R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le (cfc f a) ((algebraMap R A) r)) (∀ (x : R), Membership.mem (spect …
  -/
  rw [← cfc_const r a, cfc_le_iff ..]
  /-
    🎉 no goals
  -/


lemma algebraMap_le_cfc_iff (f : R → R) (r : R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    algebraMap R A r ≤ cfc f a ↔ ∀ x ∈ spectrum R a, r ≤ f x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.827101) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    r : R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le ((algebraMap R A) r) (cfc f a)) (∀ (x : R), Membership.mem (spect …
  -/
  rw [← cfc_const r a, cfc_le_iff ..]
  /-
    🎉 no goals
  -/


lemma le_algebraMap_iff_spectrum_le {r : R} {a : A} (ha : p a := by cfc_tac) :
    a ≤ algebraMap R A r ↔ ∀ x ∈ spectrum R a, x ≤ r := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.832368) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le a ((algebraMap R A) r)) (∀ (x : R), Membership.mem (spectrum R a) …
  -/
  nth_rw 1 [← cfc_id R a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.832368) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le (cfc id a) ((algebraMap R A) r)) (∀ (x : R), Membership.mem (spec …
  -/
  exact cfc_le_algebraMap_iff id r a
  /-
    🎉 no goals
  -/


lemma algebraMap_le_iff_le_spectrum {r : R} {a : A} (ha : p a := by cfc_tac) :
    algebraMap R A r ≤ a ↔ ∀ x ∈ spectrum R a, r ≤ x:= by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.836458) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le ((algebraMap R A) r) a) (∀ (x : R), Membership.mem (spectrum R a) …
  -/
  nth_rw 1 [← cfc_id R a]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.836458) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    r : R
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le ((algebraMap R A) r) (cfc id a)) (∀ (x : R), Membership.mem (spec …
  -/
  exact algebraMap_le_cfc_iff id r a
  /-
    🎉 no goals
  -/


lemma cfc_le_one_iff (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    cfc f a ≤ 1 ↔ ∀ x ∈ spectrum R a, f x ≤ 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.840548) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le (cfc f a) 1) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le  …
  -/
  simpa using cfc_le_algebraMap_iff f 1 a
  /-
    🎉 no goals
  -/


lemma one_le_cfc_iff (f : R → R) (a : A)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    1 ≤ cfc f a ↔ ∀ x ∈ spectrum R a, 1 ≤ f x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.850362) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    f : R → R
    a : A
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le 1 (cfc f a)) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le  …
  -/
  simpa using algebraMap_le_cfc_iff f 1 a
  /-
    🎉 no goals
  -/


lemma CFC.le_one_iff (a : A) (ha : p a := by cfc_tac) :
    a ≤ 1 ↔ ∀ x ∈ spectrum R a, x ≤ 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.860279) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le a 1) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le x 1)
  -/
  simpa using le_algebraMap_iff_spectrum_le (r := (1 : R)) (a := a)
  /-
    🎉 no goals
  -/


lemma CFC.one_le_iff (a : A) (ha : p a := by cfc_tac) :
    1 ≤ a ↔ ∀ x ∈ spectrum R a, 1 ≤ x := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝¹² : OrderedCommRing R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : MetricSpace R
    inst✝⁹ : TopologicalRing R
    inst✝⁸ : ContinuousStar R
    inst✝⁷ : ∀ (α : Type ?u.869429) [inst : TopologicalSpace α], StarOrderedRing ( …
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : PartialOrder A
    inst✝² : StarOrderedRing A
    inst✝¹ : Algebra R A
    instCFC : ContinuousFunctionalCalculus R p
    inst✝ : NonnegSpectrumClass R A
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Iff (LE.le 1 a) (∀ (x : R), Membership.mem (spectrum R a) x → LE.le 1 x)
  -/
  simpa using algebraMap_le_iff_le_spectrum (r := (1 : R)) (a := a)
  /-
    🎉 no goals
  -/


/-- The composition of `cfcHom` with the natural embedding `C(s, R) → C(spectrum R a, R)`
whenever `spectrum R a ⊆ s`.

This is sometimes necessary in order to consider the same continuous functions applied to multiple
distinct elements, with the added constraint that `cfc` does not suffice. This can occur, for
example, if it is necessary to use uniqueness of this continuous functional calculus. -/
@[simps!]
noncomputable def cfcHomSuperset {a : A} (ha : p a) {s : Set R} (hs : spectrum R a ⊆ s) :
    C(s, R) →⋆ₐ[R] A :=
  cfcHom ha |>.comp <| ContinuousMap.compStarAlgHom' R R <| ⟨_, continuous_id.subtype_map hs⟩


lemma cfcHomSuperset_continuous {a : A} (ha : p a) {s : Set R} (hs : spectrum R a ⊆ s) :
    Continuous (cfcHomSuperset ha hs) :=
  (cfcHom_continuous ha).comp <| ContinuousMap.continuous_precomp _


lemma cfcHomSuperset_id {a : A} (ha : p a) {s : Set R} (hs : spectrum R a ⊆ s) :
    cfcHomSuperset ha hs (.restrict s <| .id R) = a :=
  cfcHom_id ha


