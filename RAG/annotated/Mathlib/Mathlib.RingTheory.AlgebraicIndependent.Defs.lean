/-- `AlgebraicIndependent R x` states the family of elements `x`
  is algebraically independent over `R`, meaning that the canonical
  map out of the multivariable polynomial ring is injective. -/
def AlgebraicIndependent : Prop :=
  Injective (MvPolynomial.aeval x : MvPolynomial ι R →ₐ[R] A)


theorem algebraicIndependent_iff :
    AlgebraicIndependent R x ↔
      ∀ p : MvPolynomial ι R, MvPolynomial.aeval (x : ι → A) p = 0 → p = 0 :=
  injective_iff_map_eq_zero _


theorem AlgebraicIndependent.eq_zero_of_aeval_eq_zero (h : AlgebraicIndependent R x) :
    ∀ p : MvPolynomial ι R, MvPolynomial.aeval (x : ι → A) p = 0 → p = 0 :=
  algebraicIndependent_iff.1 h


theorem algebraicIndependent_iff_injective_aeval :
    AlgebraicIndependent R x ↔ Injective (MvPolynomial.aeval x : MvPolynomial ι R →ₐ[R] A) :=
  Iff.rfl


theorem of_comp (f : A →ₐ[R] A') (hfv : AlgebraicIndependent R (f ∘ x)) :
    AlgebraicIndependent R x := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    f : AlgHom R A A'
    hfv : AlgebraicIndependent R (Function.comp (⇑f) x)
    ⊢ AlgebraicIndependent R x
  -/
  have : aeval (f ∘ x) = f.comp (aeval x) := by ext; simp
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    f : AlgHom R A A'
    hfv : AlgebraicIndependent R (Function.comp (⇑f) x)
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x)) (f.comp (MvPolynomial.ae …
    ⊢ AlgebraicIndependent R x
  -/
  rw [AlgebraicIndependent, this, AlgHom.coe_comp] at hfv
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    f : AlgHom R A A'
    hfv : Function.Injective (Function.comp ⇑f ⇑(MvPolynomial.aeval x))
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x)) (f.comp (MvPolynomial.ae …
    ⊢ AlgebraicIndependent R x
  -/
  exact hfv.of_comp
  /-
    🎉 no goals
  -/


theorem comp (f : ι' → ι) (hf : Function.Injective f) : AlgebraicIndependent R (x ∘ f) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι' → ι
    hf : Function.Injective f
    ⊢ AlgebraicIndependent R (Function.comp x f)
  -/
  intro p q
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι' → ι
    hf : Function.Injective f
    p q : MvPolynomial ι' R
    ⊢ Eq ((MvPolynomial.aeval (Function.comp x f)) p) ((MvPolynomial.aeval (Functi …
  -/
  simpa [aeval_rename, (rename_injective f hf).eq_iff] using @hx (rename f p) (rename f q)
  /-
    🎉 no goals
  -/


theorem coe_range : AlgebraicIndependent R ((↑) : range x → A) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  simpa using hx.comp _ (rangeSplitting_injective x)
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_equiv (e : ι ≃ ι') {f : ι' → A} :
    AlgebraicIndependent R (f ∘ e) ↔ AlgebraicIndependent R f :=
  ⟨fun h => Function.comp_id f ▸ e.self_comp_symm ▸ h.comp _ e.symm.injective,
    fun h => h.comp _ e.injective⟩


theorem algebraicIndependent_equiv' (e : ι ≃ ι') {f : ι' → A} {g : ι → A} (h : f ∘ e = g) :
    AlgebraicIndependent R g ↔ AlgebraicIndependent R f :=
  h ▸ algebraicIndependent_equiv e


theorem algebraicIndependent_subtype_range {ι} {f : ι → A} (hf : Injective f) :
    AlgebraicIndependent R ((↑) : range f → A) ↔ AlgebraicIndependent R f :=
  Iff.symm <| algebraicIndependent_equiv' (Equiv.ofInjective f hf) rfl


alias ⟨AlgebraicIndependent.of_subtype_range, _⟩ := algebraicIndependent_subtype_range


theorem algebraicIndependent_image {ι} {s : Set ι} {f : ι → A} (hf : Set.InjOn f s) :
    (AlgebraicIndependent R fun x : s => f x) ↔ AlgebraicIndependent R fun x : f '' s => (x : A) :=
  algebraicIndependent_equiv' (Equiv.Set.imageOfInjOn _ _ hf) rfl


theorem mono {t s : Set A} (h : t ⊆ s)
    (hx : AlgebraicIndependent R ((↑) : s → A)) : AlgebraicIndependent R ((↑) : t → A) := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    t s : Set A
    h : HasSubset.Subset t s
    hx : AlgebraicIndependent R Subtype.val
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  simpa [Function.comp] using hx.comp (inclusion h) (inclusion_injective h)
  /-
    🎉 no goals
  -/


/-- Canonical isomorphism between polynomials and the subalgebra generated by
  algebraically independent elements. -/
@[simps! apply_coe]
def aevalEquiv : MvPolynomial ι R ≃ₐ[R] Algebra.adjoin R (range x) :=
  (AlgEquiv.ofInjective (aeval x) (algebraicIndependent_iff_injective_aeval.1 hx)).trans
    (Subalgebra.equivOfEq _ _ (Algebra.adjoin_range_eq_range_aeval R x).symm)

--@[simp] Porting note: removing simp because the linter complains about deterministic timeout

theorem algebraMap_aevalEquiv (p : MvPolynomial ι R) :
    algebraMap (Algebra.adjoin R (range x)) A (hx.aevalEquiv p) = aeval x p :=
  rfl


/-- The canonical map from the subalgebra generated by an algebraic independent family
  into the polynomial ring. -/
def repr : Algebra.adjoin R (range x) →ₐ[R] MvPolynomial ι R :=
  hx.aevalEquiv.symm


@[simp]
theorem aeval_repr (p) : aeval x (hx.repr p) = p :=
  Subtype.ext_iff.1 (AlgEquiv.apply_symm_apply hx.aevalEquiv p)


theorem aeval_comp_repr : (aeval x).comp hx.repr = Subalgebra.val _ :=
  AlgHom.ext hx.aeval_repr


/-- A family is a transcendence basis if it is a maximal algebraically independent subset. -/
def IsTranscendenceBasis (x : ι → A) : Prop :=
  AlgebraicIndependent R x ∧
    ∀ (s : Set A) (_ : AlgebraicIndependent R ((↑) : s → A)) (_ : range x ≤ s), range x = s

