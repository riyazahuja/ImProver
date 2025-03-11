/-- A basis on `S` gives a basis on `Ideal.span {x}`, by multiplying everything by `x`. -/
noncomputable def basisSpanSingleton (b : Basis ι R S) {x : S} (hx : x ≠ 0) :
    Basis ι R (span ({x} : Set S)) :=
  b.map <|
    LinearEquiv.ofInjective (LinearMap.mulLeft R x) (mul_right_injective₀ hx) ≪≫ₗ
        LinearEquiv.ofEq _ _
          (by
            /-
              ι : Type u_1
              R : Type u_2
              S : Type u_3
              inst✝³ : CommSemiring R
              inst✝² : CommRing S
              inst✝¹ : IsDomain S
              inst✝ : Algebra R S
              b : Basis ι R S
              x : S
              hx : Ne x 0
              ⊢ Eq (LinearMap.range (LinearMap.mulLeft R x)) (Submodule.restrictScalars R (I …
            -/
            ext
            /-
              case h
              ι : Type u_1
              R : Type u_2
              S : Type u_3
              inst✝³ : CommSemiring R
              inst✝² : CommRing S
              inst✝¹ : IsDomain S
              inst✝ : Algebra R S
              b : Basis ι R S
              x : S
              hx : Ne x 0
              x✝ : S
              ⊢ Iff (Membership.mem (LinearMap.range (LinearMap.mulLeft R x)) x✝) (Membershi …
            -/
            simp [mem_span_singleton', mul_comm]) ≪≫ₗ
            /-
              🎉 no goals
            -/
      (Submodule.restrictScalarsEquiv R S S (Ideal.span ({x} : Set S))).restrictScalars R


@[simp]
theorem basisSpanSingleton_apply (b : Basis ι R S) {x : S} (hx : x ≠ 0) (i : ι) :
    (basisSpanSingleton b hx i : S) = x * b i := by
  simp only [basisSpanSingleton, Basis.map_apply, LinearEquiv.trans_apply,
    Submodule.restrictScalarsEquiv_apply, LinearEquiv.ofInjective_apply, LinearEquiv.coe_ofEq_apply,
    LinearEquiv.restrictScalars_apply, LinearMap.mulLeft_apply, LinearMap.mul_apply']


@[simp]
theorem constr_basisSpanSingleton {N : Type*} [Semiring N] [Module N S] [SMulCommClass R N S]
    (b : Basis ι R S) {x : S} (hx : x ≠ 0) :
    (b.constr N).toFun (((↑) : _ → S) ∘ (basisSpanSingleton b hx)) = Algebra.lmul R S x :=
  b.ext fun i => by
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain S
      inst✝³ : Algebra R S
      N : Type u_4
      inst✝² : Semiring N
      inst✝¹ : Module N S
      inst✝ : SMulCommClass R N S
      b : Basis ι R S
      x : S
      hx : Ne x 0
      i : ι
      ⊢ Eq (((↑(b.constr N)).toFun (Function.comp Subtype.val ⇑(Ideal.basisSpanSingl …
    -/
    erw [Basis.constr_basis, Function.comp_apply, basisSpanSingleton_apply, LinearMap.mul_apply']
    /-
      🎉 no goals
    -/


/-- If `I : Ideal S` has a basis over `R`,
`x ∈ I` iff it is a linear combination of basis vectors. -/
theorem Basis.mem_ideal_iff {ι R S : Type*} [CommRing R] [CommRing S] [Algebra R S] {I : Ideal S}
    (b : Basis ι R I) {x : S} : x ∈ I ↔ ∃ c : ι →₀ R, x = Finsupp.sum c fun i x => x • (b i : S) :=
  (b.map ((I.restrictScalarsEquiv R _ _).restrictScalars R).symm).mem_submodule_iff


/-- If `I : Ideal S` has a finite basis over `R`,
`x ∈ I` iff it is a linear combination of basis vectors. -/
theorem Basis.mem_ideal_iff' {ι R S : Type*} [Fintype ι] [CommRing R] [CommRing S] [Algebra R S]
    {I : Ideal S} (b : Basis ι R I) {x : S} : x ∈ I ↔ ∃ c : ι → R, x = ∑ i, c i • (b i : S) :=
  (b.map ((I.restrictScalarsEquiv R _ _).restrictScalars R).symm).mem_submodule_iff'

