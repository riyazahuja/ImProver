theorem ker_closedComplemented_of_finiteDimensional_range (f : E →L[𝕜] F)
    [FiniteDimensional 𝕜 (range f)] : (ker f).ClosedComplemented := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    ⊢ (LinearMap.ker f).ClosedComplemented
  -/
  set f' : E →L[𝕜] range f := f.codRestrict _ (LinearMap.mem_range_self (f : E →ₗ[𝕜] F))
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    f' : ContinuousLinearMap (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (Li …
    ⊢ (LinearMap.ker f).ClosedComplemented
  -/
  rcases f'.exists_right_inverse_of_surjective (f : E →ₗ[𝕜] F).range_rangeRestrict with ⟨g, hg⟩
  simpa only [f', ker_codRestrict]
    using f'.closedComplemented_ker_of_rightInverse g (ContinuousLinearMap.ext_iff.1 hg)


/-- If `f : E →L[R] F` and `g : E →L[R] G` are two surjective linear maps and
their kernels are complement of each other, then `x ↦ (f x, g x)` defines
a linear equivalence `E ≃L[R] F × G`. -/
nonrec def equivProdOfSurjectiveOfIsCompl (f : E →L[𝕜] F) (g : E →L[𝕜] G) (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) : E ≃L[𝕜] F × G :=
  (f.equivProdOfSurjectiveOfIsCompl (g : E →ₗ[𝕜] G) hf hg hfg).toContinuousLinearEquivOfContinuous
    (f.continuous.prod_mk g.continuous)


@[simp]
theorem coe_equivProdOfSurjectiveOfIsCompl {f : E →L[𝕜] F} {g : E →L[𝕜] G} (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) :
    (equivProdOfSurjectiveOfIsCompl f g hf hg hfg : E →ₗ[𝕜] F × G) = f.prod g := rfl


@[simp]
theorem equivProdOfSurjectiveOfIsCompl_toLinearEquiv {f : E →L[𝕜] F} {g : E →L[𝕜] G}
    (hf : range f = ⊤) (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) :
    (equivProdOfSurjectiveOfIsCompl f g hf hg hfg).toLinearEquiv =
      LinearMap.equivProdOfSurjectiveOfIsCompl f g hf hg hfg := rfl


@[simp]
theorem equivProdOfSurjectiveOfIsCompl_apply {f : E →L[𝕜] F} {g : E →L[𝕜] G} (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) (x : E) :
    equivProdOfSurjectiveOfIsCompl f g hf hg hfg x = (f x, g x) := rfl


/-- If `q` is a closed complement of a closed subspace `p`, then `p × q` is continuously
isomorphic to `E`. -/
def prodEquivOfClosedCompl (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) : (p × q) ≃L[𝕜] E := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace E
    p q : Subspace 𝕜 E
    h : IsCompl p q
    hp : IsClosed ↑p
    hq : IsClosed ↑q
    ⊢ ContinuousLinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem  …
  -/
  haveI := hp.completeSpace_coe; haveI := hq.completeSpace_coe
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace E
    p q : Subspace 𝕜 E
    h : IsCompl p q
    hp : IsClosed ↑p
    hq : IsClosed ↑q
    this✝ : CompleteSpace ↑↑p
    this : CompleteSpace ↑↑q
    ⊢ ContinuousLinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem  …
  -/
  refine (p.prodEquivOfIsCompl q h).toContinuousLinearEquivOfContinuous ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace E
    p q : Subspace 𝕜 E
    h : IsCompl p q
    hp : IsClosed ↑p
    hq : IsClosed ↑q
    this✝ : CompleteSpace ↑↑p
    this : CompleteSpace ↑↑q
    ⊢ Continuous ⇑(Submodule.prodEquivOfIsCompl p q h)
  -/
  exact (p.subtypeL.coprod q.subtypeL).continuous
  /-
    🎉 no goals
  -/


/-- Projection to a closed submodule along a closed complement. -/
def linearProjOfClosedCompl (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) : E →L[𝕜] p :=
  ContinuousLinearMap.fst 𝕜 p q ∘L ↑(prodEquivOfClosedCompl p q h hp hq).symm


@[simp]
theorem coe_prodEquivOfClosedCompl (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) :
    ⇑(p.prodEquivOfClosedCompl q h hp hq) = p.prodEquivOfIsCompl q h := rfl


@[simp]
theorem coe_prodEquivOfClosedCompl_symm (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) :
    ⇑(p.prodEquivOfClosedCompl q h hp hq).symm = (p.prodEquivOfIsCompl q h).symm := rfl


@[simp]
theorem coe_continuous_linearProjOfClosedCompl (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) :
    (p.linearProjOfClosedCompl q h hp hq : E →ₗ[𝕜] p) = p.linearProjOfIsCompl q h := rfl


@[simp]
theorem coe_continuous_linearProjOfClosedCompl' (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) :
    ⇑(p.linearProjOfClosedCompl q h hp hq) = p.linearProjOfIsCompl q h := rfl


theorem ClosedComplemented.of_isCompl_isClosed (h : IsCompl p q) (hp : IsClosed (p : Set E))
    (hq : IsClosed (q : Set E)) : p.ClosedComplemented :=
  ⟨p.linearProjOfClosedCompl q h hp hq, Submodule.linearProjOfIsCompl_apply_left h⟩


alias IsCompl.closedComplemented_of_isClosed := ClosedComplemented.of_isCompl_isClosed


theorem closedComplemented_iff_isClosed_exists_isClosed_isCompl :
    p.ClosedComplemented ↔
      IsClosed (p : Set E) ∧ ∃ q : Submodule 𝕜 E, IsClosed (q : Set E) ∧ IsCompl p q :=
  ⟨fun h => ⟨h.isClosed, h.exists_isClosed_isCompl⟩,
    fun ⟨hp, ⟨_, hq, hpq⟩⟩ => .of_isCompl_isClosed hpq hp hq⟩


theorem ClosedComplemented.of_quotient_finiteDimensional [CompleteSpace 𝕜]
    [FiniteDimensional 𝕜 (E ⧸ p)] (hp : IsClosed (p : Set E)) : p.ClosedComplemented := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompleteSpace E
    p : Subspace 𝕜 E
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    hp : IsClosed ↑p
    ⊢ Submodule.ClosedComplemented p
  -/
  obtain ⟨q, hq⟩ : ∃ q, IsCompl p q := p.exists_isCompl
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompleteSpace E
    p : Subspace 𝕜 E
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    hp : IsClosed ↑p
    q : Subspace 𝕜 E
    hq : IsCompl p q
    ⊢ Submodule.ClosedComplemented p
  -/
  haveI : FiniteDimensional 𝕜 q := (p.quotientEquivOfIsCompl q hq).finiteDimensional
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : CompleteSpace E
    p : Subspace 𝕜 E
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    hp : IsClosed ↑p
    q : Subspace 𝕜 E
    hq : IsCompl p q
    this : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem q x)
    ⊢ Submodule.ClosedComplemented p
  -/
  exact .of_isCompl_isClosed hq hp q.closed_of_finiteDimensional
  /-
    🎉 no goals
  -/


