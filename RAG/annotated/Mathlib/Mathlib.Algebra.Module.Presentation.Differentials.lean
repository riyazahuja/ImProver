/-- The shape of the presentation by generators and relations of the `S`-module `Ω[S⁄R]`
that is obtained from a presentation of `S` as an `R`-algebra. -/
@[simps G R]
noncomputable def differentialsRelations : Module.Relations S where
  G := pres.vars
  R := pres.rels
  relation r :=
                                                  /-
                                                    R : Type u
                                                    S : Type v
                                                    inst✝² : CommRing R
                                                    inst✝¹ : CommRing S
                                                    inst✝ : Algebra R S
                                                    pres : Algebra.Presentation R S
                                                    r : pres.rels
                                                    ⊢ Eq ((algebraMap pres.Ring S) 0) 0
                                                  -/
    Finsupp.mapRange (algebraMap pres.Ring S) (by simp)
                                                  /-
                                                    🎉 no goals
                                                  -/
      ((mvPolynomialBasis R pres.vars).repr (D _ _ (pres.relation r)))


/-- Same as `comm₂₃` below, but here we have not yet constructed `differentialsSolution`. -/
lemma comm₂₃' : pres.toExtension.toKaehler.comp pres.cotangentSpaceBasis.repr.symm.toLinearMap =
    Finsupp.linearCombination S (fun g ↦ D _ _ (pres.val g)) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    ⊢ Eq (pres.toExtension.toKaehler.comp ↑pres.cotangentSpaceBasis.repr.symm) (Fi …
  -/
  ext g
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    g : pres.vars
    ⊢ Eq (((pres.toExtension.toKaehler.comp ↑pres.cotangentSpaceBasis.repr.symm).c …
  -/
  dsimp
  rw [Basis.repr_symm_apply, Finsupp.linearCombination_single,
    Finsupp.linearCombination_single, one_smul, one_smul,
    Generators.cotangentSpaceBasis_apply]
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    g : pres.vars
    ⊢ Eq (pres.toExtension.toKaehler (TensorProduct.tmul pres.Ring 1 ((KaehlerDiff …
  -/
  simp [Generators.toExtension]
  /-
    🎉 no goals
  -/


/-- The canonical map `(pres.rels →₀ S) →ₗ[S] pres.toExtension.Cotangent`. -/
noncomputable def hom₁ : (pres.rels →₀ S) →ₗ[S] pres.toExtension.Cotangent :=
                                                                                   /-
                                                                                     R : Type u
                                                                                     S : Type v
                                                                                     inst✝² : CommRing R
                                                                                     inst✝¹ : CommRing S
                                                                                     inst✝ : Algebra R S
                                                                                     pres : Algebra.Presentation R S
                                                                                     r : pres.rels
                                                                                     ⊢ Membership.mem pres.toExtension.ker (pres.relation r)
                                                                                   -/
  Finsupp.linearCombination S (fun r ↦ Extension.Cotangent.mk ⟨pres.relation r, by simp⟩)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma hom₁_single (r : pres.rels) :
                                                                                 /-
                                                                                   R : Type u
                                                                                   S : Type v
                                                                                   inst✝² : CommRing R
                                                                                   inst✝¹ : CommRing S
                                                                                   inst✝ : Algebra R S
                                                                                   pres : Algebra.Presentation R S
                                                                                   r : pres.rels
                                                                                   ⊢ Membership.mem pres.toExtension.ker (pres.relation r)
                                                                                 -/
    hom₁ pres (Finsupp.single r 1) = Extension.Cotangent.mk ⟨pres.relation r, by simp⟩ := by
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    r : pres.rels
    ⊢ Eq ((Algebra.Presentation.differentials.hom₁ pres) (Finsupp.single r 1)) (Al …
  -/
  simp [hom₁]
  /-
    🎉 no goals
  -/


lemma surjective_hom₁ : Function.Surjective (hom₁ pres) := by
  let φ : (pres.rels →₀ S) →ₗ[pres.Ring] pres.toExtension.Cotangent :=
    { toFun := hom₁ pres
      map_add' := by simp
      map_smul' := by simp }
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    ⊢ Function.Surjective ⇑(Algebra.Presentation.differentials.hom₁ pres)
  -/
  change Function.Surjective φ
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    ⊢ Function.Surjective ⇑φ
  -/
  have h₁ := Algebra.Extension.Cotangent.mk_surjective (P := pres.toExtension)
  have h₂ : Submodule.span pres.Ring
      (Set.range (fun r ↦ (⟨pres.relation r, by simp⟩ : pres.ker))) = ⊤ := by
    refine Submodule.map_injective_of_injective (f := Submodule.subtype pres.ker)
      Subtype.coe_injective ?_
    rw [Submodule.map_top, Submodule.range_subtype, Submodule.map_span,
      Submodule.coe_subtype, Ideal.submodule_span_eq]
    simp only [← pres.span_range_relation_eq_ker]
    congr
    aesop
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Function.Surjective ⇑Algebra.Extension.Cotangent.mk
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    ⊢ Function.Surjective ⇑φ
  -/
  rw [← LinearMap.range_eq_top] at h₁ ⊢
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    ⊢ Eq (LinearMap.range φ) Top.top
  -/
  rw [← top_le_iff, ← h₁, LinearMap.range_eq_map, ← h₂]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    ⊢ LE.le (Submodule.map Algebra.Extension.Cotangent.mk (Submodule.span pres.Rin …
  -/
  dsimp
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    ⊢ LE.le (Submodule.map Algebra.Extension.Cotangent.mk (Submodule.span pres.Rin …
  -/
  rw [Submodule.map_span_le]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    ⊢ ∀ (m : Subtype fun x => Membership.mem pres.toExtension.ker x), Membership.m …
  -/
  rintro _ ⟨r, rfl⟩
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    r : pres.rels
    ⊢ Membership.mem (LinearMap.range φ) (Algebra.Extension.Cotangent.mk ((fun r = …
  -/
  simp only [LinearMap.mem_range]
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    r : pres.rels
    ⊢ Exists fun y => Eq (φ y) (Algebra.Extension.Cotangent.mk ⟨pres.relation r, ⋯⟩)
  -/
  refine ⟨Finsupp.single r 1, ?_⟩
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    r : pres.rels
    ⊢ Eq (φ (Finsupp.single r 1)) (Algebra.Extension.Cotangent.mk ⟨pres.relation r …
  -/
  simp only [LinearMap.coe_mk, AddHom.coe_mk, hom₁_single, φ]
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    φ : LinearMap (RingHom.id pres.Ring) (Finsupp pres.rels S) pres.toExtension.Co …
    h₁ : Eq (LinearMap.range Algebra.Extension.Cotangent.mk) Top.top
    h₂ : Eq (Submodule.span pres.Ring (Set.range fun r => ⟨pres.relation r, ⋯⟩)) T …
    r : pres.rels
    ⊢ Eq (Algebra.Extension.Cotangent.mk ⟨pres.relation r, ⋯⟩) (Algebra.Extension. …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma comm₁₂_single (r : pres.rels) :
    pres.toExtension.cotangentComplex (hom₁ pres (Finsupp.single r 1)) =
      pres.cotangentSpaceBasis.repr.symm ((differentialsRelations pres).relation r) := by
  simp only [hom₁, Finsupp.linearCombination_single, one_smul, differentialsRelations,
    Basis.repr_symm_apply, Extension.cotangentComplex_mk]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    r : pres.rels
    ⊢ Eq (TensorProduct.tmul pres.toExtension.Ring 1 ((KaehlerDifferential.D R pre …
  -/
  exact pres.cotangentSpaceBasis.repr.injective (by ext; simp)
  /-
    🎉 no goals
  -/


lemma comm₁₂ : pres.toExtension.cotangentComplex.comp (hom₁ pres) =
    pres.cotangentSpaceBasis.repr.symm.comp (differentialsRelations pres).map := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    ⊢ Eq (pres.toExtension.cotangentComplex.comp (Algebra.Presentation.differentia …
  -/
  ext r
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    r : pres.rels
    ⊢ Eq (((pres.toExtension.cotangentComplex.comp (Algebra.Presentation.different …
  -/
  have := (differentialsRelations pres).map_single
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    r : pres.rels
    this : ∀ (r : pres.differentialsRelations.R), Eq (pres.differentialsRelations. …
    ⊢ Eq (((pres.toExtension.cotangentComplex.comp (Algebra.Presentation.different …
  -/
  dsimp at this ⊢
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    r : pres.rels
    this : ∀ (r : pres.rels), Eq (pres.differentialsRelations.map (Finsupp.single  …
    ⊢ Eq (pres.toExtension.cotangentComplex ((Algebra.Presentation.differentials.h …
  -/
  rw [comm₁₂_single, this]
  /-
    🎉 no goals
  -/


open differentials in
/-- The `S`-module `Ω[S⁄R]` contains an obvious solution to the system of linear
equations `pres.differentialsRelations.Solution` when `pres` is a presentation
of `S` as an `R`-algebra. -/
noncomputable def differentialsSolution :
    pres.differentialsRelations.Solution (Ω[S⁄R]) where
  var g := D _ _ (pres.val g)
  linearCombination_var_relation r := by
    simp only [differentialsRelations_G, LinearMap.coe_comp, LinearEquiv.coe_coe,
      Function.comp_apply, ← comm₂₃', ← comm₁₂_single]
    apply DFunLike.congr_fun (Function.Exact.linearMap_comp_eq_zero
      (pres.toExtension.exact_cotangentComplex_toKaehler))


lemma differentials.comm₂₃ :
    pres.toExtension.toKaehler.comp pres.cotangentSpaceBasis.repr.symm.toLinearMap =
      pres.differentialsSolution.π :=
  comm₂₃' pres


open differentials in
lemma differentialsSolution_isPresentation :
    pres.differentialsSolution.IsPresentation := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    ⊢ pres.differentialsSolution.IsPresentation
  -/
  rw [Module.Relations.Solution.isPresentation_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    pres : Algebra.Presentation R S
    ⊢ And (Eq (Submodule.span S (Set.range pres.differentialsSolution.var)) Top.to …
  -/
  constructor
    /-
      case left
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      pres : Algebra.Presentation R S
      ⊢ Eq (Submodule.span S (Set.range pres.differentialsSolution.var)) Top.top
    -/
  · rw [← Module.Relations.Solution.surjective_π_iff_span_eq_top, ← comm₂₃]
    /-
      case left
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      pres : Algebra.Presentation R S
      ⊢ Function.Surjective ⇑(pres.toExtension.toKaehler.comp ↑pres.cotangentSpaceBa …
    -/
    exact Extension.toKaehler_surjective.comp pres.cotangentSpaceBasis.repr.symm.surjective
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      pres : Algebra.Presentation R S
      ⊢ Eq (LinearMap.ker pres.differentialsSolution.π) (Submodule.span S (Set.range …
    -/
  · rw [← Module.Relations.range_map]
    exact Function.Exact.linearMap_ker_eq
      ((LinearMap.exact_iff_of_surjective_of_bijective_of_injective
      _ _ _ _ (hom₁ pres)
      pres.cotangentSpaceBasis.repr.symm.toLinearMap .id
      (comm₁₂ pres) (by simpa using comm₂₃ pres) (surjective_hom₁ pres)
        (LinearEquiv.bijective _) (Equiv.refl _).injective).2
        pres.toExtension.exact_cotangentComplex_toKaehler)


/-- The presentation of the `S`-module `Ω[S⁄R]` deduced from a presentation
of `S` as a `R`-algebra. -/
noncomputable def differentials : Module.Presentation S (Ω[S⁄R]) where
  toSolution := differentialsSolution pres
  toIsPresentation := pres.differentialsSolution_isPresentation


