/-- Suppose that `R → S → A` is a tower of algebras.
If an element `r : R` is invertible in `S`, then it is invertible in `A`. -/
def Invertible.algebraTower (r : R) [Invertible (algebraMap R S r)] :
    Invertible (algebraMap R A r) :=
  Invertible.copy (Invertible.map (algebraMap S A) (algebraMap R S r)) (algebraMap R A r)
    (IsScalarTower.algebraMap_apply R S A r)


/-- A natural number that is invertible when coerced to `R` is also invertible
when coerced to any `R`-algebra. -/
def invertibleAlgebraCoeNat (n : ℕ) [inv : Invertible (n : R)] : Invertible (n : A) :=
  haveI : Invertible (algebraMap ℕ R n) := inv
  Invertible.algebraTower ℕ R A n


/-- If `R` and `A` have a bijective `algebraMap R A` and act identically on `M`,
then a basis for `M` as `R`-module is also a basis for `M` as `R'`-module. -/
@[simps! repr_apply_support_val repr_apply_toFun]
noncomputable def Basis.algebraMapCoeffs : Basis ι A M :=
                                                        /-
                                                          R : Type u
                                                          S : Type v
                                                          A : Type w
                                                          B : Type u₁
                                                          ι : Type u_1
                                                          M : Type u_2
                                                          inst✝⁶ : CommSemiring R
                                                          inst✝⁵ : Semiring A
                                                          inst✝⁴ : AddCommMonoid M
                                                          inst✝³ : Algebra R A
                                                          inst✝² : Module A M
                                                          inst✝¹ : Module R M
                                                          inst✝ : IsScalarTower R A M
                                                          b : Basis ι R M
                                                          h : Function.Bijective ⇑(algebraMap R A)
                                                          c : R
                                                          x : M
                                                          ⊢ Eq (HSMul.hSMul ((RingEquiv.ofBijective (algebraMap R A) h) c) x) (HSMul.hSM …
                                                        -/
  b.mapCoeffs (RingEquiv.ofBijective _ h) fun c x => by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem Basis.algebraMapCoeffs_apply (i : ι) : b.algebraMapCoeffs A h i = b i :=
  b.mapCoeffs_apply _ _ _


@[simp]
theorem Basis.coe_algebraMapCoeffs : (b.algebraMapCoeffs A h : ι → M) = b :=
  b.coe_mapCoeffs _ _


theorem linearIndependent_smul {ι : Type v₁} {b : ι → S} {ι' : Type w₁} {c : ι' → A}
    (hb : LinearIndependent R b) (hc : LinearIndependent S c) :
    LinearIndependent R fun p : ι × ι' => b p.1 • c p.2 := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Ring R
    inst✝⁵ : Ring S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type v₁
    b : ι → S
    ι' : Type w₁
    c : ι' → A
    hb : LinearIndependent R b
    hc : LinearIndependent S c
    ⊢ LinearIndependent R fun p => HSMul.hSMul (b p.1) (c p.2)
  -/
  rw [linearIndependent_iff'] at hb hc; rw [linearIndependent_iff'']; rintro s g hg hsg ⟨i, k⟩
  /-
    case mk
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Ring R
    inst✝⁵ : Ring S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type v₁
    b : ι → S
    ι' : Type w₁
    c : ι' → A
    hb : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (b i)) …
    hc : ∀ (s : Finset ι') (g : ι' → S), Eq (s.sum fun i => HSMul.hSMul (g i) (c i …
    s : Finset (Prod ι ι')
    g : Prod ι ι' → R
    hg : ∀ (i : Prod ι ι'), Not (Membership.mem s i) → Eq (g i) 0
    hsg : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul (b i.1) (c i.2))) 0
    i : ι
    k : ι'
    ⊢ Eq (g { fst := i, snd := k }) 0
  -/
  by_cases hik : (i, k) ∈ s
  · have h1 : ∑ i ∈ s.image Prod.fst ×ˢ s.image Prod.snd, g i • b i.1 • c i.2 = 0 := by
      rw [← hsg]
      exact
        (Finset.sum_subset Finset.subset_product fun p _ hp =>
            show g p • b p.1 • c p.2 = 0 by rw [hg p hp, zero_smul]).symm
    /-
      case pos
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R S
      inst✝² : Module S A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R S A
      ι : Type v₁
      b : ι → S
      ι' : Type w₁
      c : ι' → A
      hb : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (b i)) …
      hc : ∀ (s : Finset ι') (g : ι' → S), Eq (s.sum fun i => HSMul.hSMul (g i) (c i …
      s : Finset (Prod ι ι')
      g : Prod ι ι' → R
      hg : ∀ (i : Prod ι ι'), Not (Membership.mem s i) → Eq (g i) 0
      hsg : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul (b i.1) (c i.2))) 0
      i : ι
      k : ι'
      hik : Membership.mem s { fst := i, snd := k }
      h1 : Eq ((SProd.sprod (Finset.image Prod.fst s) (Finset.image Prod.snd s)).sum …
      ⊢ Eq (g { fst := i, snd := k }) 0
    -/
    rw [Finset.sum_product_right] at h1
    /-
      case pos
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R S
      inst✝² : Module S A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R S A
      ι : Type v₁
      b : ι → S
      ι' : Type w₁
      c : ι' → A
      hb : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (b i)) …
      hc : ∀ (s : Finset ι') (g : ι' → S), Eq (s.sum fun i => HSMul.hSMul (g i) (c i …
      s : Finset (Prod ι ι')
      g : Prod ι ι' → R
      hg : ∀ (i : Prod ι ι'), Not (Membership.mem s i) → Eq (g i) 0
      hsg : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul (b i.1) (c i.2))) 0
      i : ι
      k : ι'
      hik : Membership.mem s { fst := i, snd := k }
      h1 : Eq ((Finset.image Prod.snd s).sum fun y => (Finset.image Prod.fst s).sum  …
      ⊢ Eq (g { fst := i, snd := k }) 0
    -/
    simp_rw [← smul_assoc, ← Finset.sum_smul] at h1
    /-
      case pos
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R S
      inst✝² : Module S A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R S A
      ι : Type v₁
      b : ι → S
      ι' : Type w₁
      c : ι' → A
      hb : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (b i)) …
      hc : ∀ (s : Finset ι') (g : ι' → S), Eq (s.sum fun i => HSMul.hSMul (g i) (c i …
      s : Finset (Prod ι ι')
      g : Prod ι ι' → R
      hg : ∀ (i : Prod ι ι'), Not (Membership.mem s i) → Eq (g i) 0
      hsg : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul (b i.1) (c i.2))) 0
      i : ι
      k : ι'
      hik : Membership.mem s { fst := i, snd := k }
      h1 : Eq ((Finset.image Prod.snd s).sum fun x => HSMul.hSMul ((Finset.image Pro …
      ⊢ Eq (g { fst := i, snd := k }) 0
    -/
    exact hb _ _ (hc _ _ h1 k (Finset.mem_image_of_mem _ hik)) i (Finset.mem_image_of_mem _ hik)
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Ring R
    inst✝⁵ : Ring S
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type v₁
    b : ι → S
    ι' : Type w₁
    c : ι' → A
    hb : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (b i)) …
    hc : ∀ (s : Finset ι') (g : ι' → S), Eq (s.sum fun i => HSMul.hSMul (g i) (c i …
    s : Finset (Prod ι ι')
    g : Prod ι ι' → R
    hg : ∀ (i : Prod ι ι'), Not (Membership.mem s i) → Eq (g i) 0
    hsg : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul (b i.1) (c i.2))) 0
    i : ι
    k : ι'
    hik : Not (Membership.mem s { fst := i, snd := k })
    ⊢ Eq (g { fst := i, snd := k }) 0
  -/
  exact hg _ hik
  /-
    🎉 no goals
  -/

theorem Basis.isScalarTower_of_nonempty {ι} [Nonempty ι] (b : Basis ι S A) : IsScalarTower R S S :=
  (b.repr.symm.comp <| lsingle <| Classical.arbitrary ι).isScalarTower_of_injective R
    (b.repr.symm.injective.comp <| single_injective _)


theorem Basis.isScalarTower_finsupp {ι} (b : Basis ι S A) : IsScalarTower R S (ι →₀ S) :=
  b.repr.symm.isScalarTower_of_injective R b.repr.symm.injective


/-- `Basis.smulTower (b : Basis ι R S) (c : Basis ι S A)` is the `R`-basis on `A`
where the `(i, j)`th basis vector is `b i • c j`. -/
noncomputable
def Basis.smulTower : Basis (ι × ι') R A :=
  haveI := c.isScalarTower_finsupp R
  .ofRepr
    (c.repr.restrictScalars R ≪≫ₗ
      (Finsupp.lcongr (Equiv.refl _) b.repr ≪≫ₗ
        ((finsuppProdLEquiv R).symm ≪≫ₗ
          Finsupp.lcongr (Equiv.prodComm ι' ι) (LinearEquiv.refl _ _))))


@[simp]
theorem Basis.smulTower_repr (x ij) :
    (b.smulTower c).repr x ij = b.repr (c.repr x ij.2) ij.1 := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    x : A
    ij : Prod ι ι'
    ⊢ Eq (((b.smulTower c).repr x) ij) ((b.repr ((c.repr x) ij.2)) ij.1)
  -/
  simp [smulTower]
  /-
    🎉 no goals
  -/


theorem Basis.smulTower_repr_mk (x i j) : (b.smulTower c).repr x (i, j) = b.repr (c.repr x j) i :=
  b.smulTower_repr c x (i, j)


@[simp]
theorem Basis.smulTower_apply (ij) : (b.smulTower c) ij = b ij.1 • c ij.2 := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    ij : Prod ι ι'
    ⊢ Eq ((b.smulTower c) ij) (HSMul.hSMul (b ij.1) (c ij.2))
  -/
  obtain ⟨i, j⟩ := ij
  /-
    case mk
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    i : ι
    j : ι'
    ⊢ Eq ((b.smulTower c) { fst := i, snd := j }) (HSMul.hSMul (b { fst := i, snd  …
  -/
  rw [Basis.apply_eq_iff]
  /-
    case mk
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    i : ι
    j : ι'
    ⊢ Eq ((b.smulTower c).repr (HSMul.hSMul (b { fst := i, snd := j }.1) (c { fst  …
  -/
  ext ⟨i', j'⟩
  rw [Basis.smulTower_repr, LinearEquiv.map_smul, Basis.repr_self, Finsupp.smul_apply,
    Finsupp.single_apply]
  /-
    case mk.h.mk
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    i : ι
    j : ι'
    i' : ι
    j' : ι'
    ⊢ Eq ((b.repr (HSMul.hSMul (b { fst := i, snd := j }.1) (ite (Eq { fst := i, s …
  -/
  dsimp only
  /-
    case mk.h.mk
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    i : ι
    j : ι'
    i' : ι
    j' : ι'
    ⊢ Eq ((b.repr (HSMul.hSMul (b i) (ite (Eq j j') 1 0))) i') ((Finsupp.single {  …
  -/
  split_ifs with hi
    /-
      case pos
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring S
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R S
      inst✝² : Module S A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R S A
      ι : Type u_1
      ι' : Type u_2
      b : Basis ι R S
      c : Basis ι' S A
      i : ι
      j : ι'
      i' : ι
      j' : ι'
      hi : Eq j j'
      ⊢ Eq ((b.repr (HSMul.hSMul (b i) 1)) i') ((Finsupp.single { fst := i, snd := j …
    -/
  · simp [hi, Finsupp.single_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      S : Type v
      A : Type w
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring S
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R S
      inst✝² : Module S A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R S A
      ι : Type u_1
      ι' : Type u_2
      b : Basis ι R S
      c : Basis ι' S A
      i : ι
      j : ι'
      i' : ι
      j' : ι'
      hi : Not (Eq j j')
      ⊢ Eq ((b.repr (HSMul.hSMul (b i) 0)) i') ((Finsupp.single { fst := i, snd := j …
    -/
  · simp [hi]
    /-
      🎉 no goals
    -/


/-- `Basis.smulTower (b : Basis ι R S) (c : Basis ι S A)` is the `R`-basis on `A`
where the `(i, j)`th basis vector is `b j • c i`. -/
noncomputable def Basis.smulTower' : Basis (ι' × ι) R A :=
  (b.smulTower c).reindex (.prodComm ..)


theorem Basis.smulTower'_repr (x ij) :
    (b.smulTower' c).repr x ij = b.repr (c.repr x ij.1) ij.2 := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    x : A
    ij : Prod ι' ι
    ⊢ Eq (((b.smulTower' c).repr x) ij) ((b.repr ((c.repr x) ij.1)) ij.2)
  -/
  rw [smulTower', repr_reindex_apply, smulTower_repr]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Basis.smulTower'_repr_mk (x i j) : (b.smulTower' c).repr x (i, j) = b.repr (c.repr x i) j :=
  b.smulTower'_repr c x (i, j)


theorem Basis.smulTower'_apply (ij) : b.smulTower' c ij = b ij.2 • c ij.1 := by
  /-
    R : Type u
    S : Type v
    A : Type w
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : AddCommMonoid A
    inst✝³ : Module R S
    inst✝² : Module S A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    ι : Type u_1
    ι' : Type u_2
    b : Basis ι R S
    c : Basis ι' S A
    ij : Prod ι' ι
    ⊢ Eq ((b.smulTower' c) ij) (HSMul.hSMul (b ij.2) (c ij.1))
  -/
  rw [smulTower', reindex_apply, smulTower_apply]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Basis.algebraMap_injective {ι : Type*} [NoZeroDivisors R] [Nontrivial S]
    (b : @Basis ι R S _ _ Algebra.toModule) : Function.Injective (algebraMap R S) :=
  have : NoZeroSMulDivisors R S := b.noZeroSMulDivisors
  NoZeroSMulDivisors.algebraMap_injective R S


/-- Restrict the domain of an `AlgHom`. -/
def AlgHom.restrictDomain : B →ₐ[A] D :=
  f.comp (IsScalarTower.toAlgHom A B C)

-- Porting note: definition below used to be
--  { f with commutes' := fun _ => rfl }
-- but it complains about not finding (Algebra B D), despite it being given in the header of the thm


/-- Extend the scalars of an `AlgHom`. -/
def AlgHom.extendScalars : @AlgHom B C D _ _ _ _ (f.restrictDomain B).toRingHom.toAlgebra where
  toFun := f.toFun
  map_one' := by simp only [toRingHom_eq_coe, RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe,
    map_one]
  map_mul' := by simp only [toRingHom_eq_coe, RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe,
    MonoidHom.toOneHom_coe, map_mul, MonoidHom.coe_coe, RingHom.coe_coe, forall_const]
  map_zero' := by simp only [toRingHom_eq_coe, RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe,
    MonoidHom.toOneHom_coe, MonoidHom.coe_coe, map_zero]
  map_add' := by simp only [toRingHom_eq_coe, RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe,
    MonoidHom.toOneHom_coe, MonoidHom.coe_coe, map_add, RingHom.coe_coe, forall_const]
  commutes' := fun _ ↦ rfl
  __ := (f.restrictDomain B).toRingHom.toAlgebra


/-- `AlgHom`s from the top of a tower are equivalent to a pair of `AlgHom`s. -/
def algHomEquivSigma :
    (C →ₐ[A] D) ≃ Σf : B →ₐ[A] D, @AlgHom B C D _ _ _ _ f.toRingHom.toAlgebra where
  toFun f := ⟨f.restrictDomain B, f.extendScalars B⟩
  invFun fg :=
    let _ := fg.1.toRingHom.toAlgebra
    fg.2.restrictScalars A
  left_inv f := by
    /-
      R : Type u
      S : Type v
      A : Type w
      B : Type u₁
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CommSemiring A
      inst✝⁷ : CommSemiring C
      inst✝⁶ : CommSemiring D
      inst✝⁵ : Algebra A C
      inst✝⁴ : Algebra A D
      inst✝³ : CommSemiring B
      inst✝² : Algebra A B
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      f✝ f : AlgHom A C D
      ⊢ Eq
          ((fun fg =>
              let x := fg.fst.toAlgebra;
              AlgHom.restrictScalars A fg.snd)
            ((fun f => ⟨AlgHom.restrictDomain B f, AlgHom.extendScalars B f⟩) f))
          f
    -/
    dsimp only
    /-
      R : Type u
      S : Type v
      A : Type w
      B : Type u₁
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CommSemiring A
      inst✝⁷ : CommSemiring C
      inst✝⁶ : CommSemiring D
      inst✝⁵ : Algebra A C
      inst✝⁴ : Algebra A D
      inst✝³ : CommSemiring B
      inst✝² : Algebra A B
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      f✝ f : AlgHom A C D
      ⊢ Eq (AlgHom.restrictScalars A (AlgHom.extendScalars B f)) f
    -/
    ext
    /-
      case H
      R : Type u
      S : Type v
      A : Type w
      B : Type u₁
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CommSemiring A
      inst✝⁷ : CommSemiring C
      inst✝⁶ : CommSemiring D
      inst✝⁵ : Algebra A C
      inst✝⁴ : Algebra A D
      inst✝³ : CommSemiring B
      inst✝² : Algebra A B
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      f✝ f : AlgHom A C D
      x✝ : C
      ⊢ Eq ((AlgHom.restrictScalars A (AlgHom.extendScalars B f)) x✝) (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u
      S : Type v
      A : Type w
      B : Type u₁
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CommSemiring A
      inst✝⁷ : CommSemiring C
      inst✝⁶ : CommSemiring D
      inst✝⁵ : Algebra A C
      inst✝⁴ : Algebra A D
      inst✝³ : CommSemiring B
      inst✝² : Algebra A B
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      f : AlgHom A C D
      ⊢ Function.RightInverse
          (fun fg =>
            let x := fg.fst.toAlgebra;
            AlgHom.restrictScalars A fg.snd)
          fun f => ⟨AlgHom.restrictDomain B f, AlgHom.extendScalars B f⟩
    -/
    rintro ⟨⟨⟨⟨⟨f, _⟩, _⟩, _⟩, _⟩, ⟨⟨⟨⟨g, _⟩, _⟩, _⟩, hg⟩⟩
    obtain rfl : f = fun x => g (algebraMap B C x) := by
      ext x
      exact (hg x).symm
    /-
      case mk.mk.mk.mk.mk.mk.mk.mk.mk
      R : Type u
      S : Type v
      A : Type w
      B : Type u₁
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CommSemiring A
      inst✝⁷ : CommSemiring C
      inst✝⁶ : CommSemiring D
      inst✝⁵ : Algebra A C
      inst✝⁴ : Algebra A D
      inst✝³ : CommSemiring B
      inst✝² : Algebra A B
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      f : AlgHom A C D
      g : C → D
      map_one'✝¹ : Eq (g 1) 1
      map_mul'✝¹ : ∀ (x y : C), Eq ({ toFun := g, map_one' := map_one'✝¹ }.toFun (HM …
      map_zero'✝¹ : Eq ((↑{ toFun := g, map_one' := map_one'✝¹, map_mul' := map_mul' …
      map_add'✝¹ : ∀ (x y : C), Eq ((↑{ toFun := g, map_one' := map_one'✝¹, map_mul' …
      map_one'✝ : Eq ((fun x => g ((algebraMap B C) x)) 1) 1
      map_mul'✝ : ∀ (x y : B), Eq ({ toFun := fun x => g ((algebraMap B C) x), map_o …
      map_zero'✝ : Eq ((↑{ toFun := fun x => g ((algebraMap B C) x), map_one' := map …
      map_add'✝ : ∀ (x y : B), Eq ((↑{ toFun := fun x => g ((algebraMap B C) x), map …
      commutes'✝ : ∀ (r : A), Eq ((↑↑{ toFun := fun x => g ((algebraMap B C) x), map …
      hg : ∀ (r : B), Eq ((↑↑{ toFun := g, map_one' := map_one'✝¹, map_mul' := map_m …
      ⊢ Eq
          ((fun f => ⟨AlgHom.restrictDomain B f, AlgHom.extendScalars B f⟩)
            ((fun fg =>
                let x := fg.fst.toAlgebra;
                AlgHom.restrictScalars A fg.snd)
              ⟨{ toFun := fun x => g ((algebraMap B C) x), map_one' := map_one'✝, ma …
          ⟨{ toFun := fun x => g ((algebraMap B C) x), map_one' := map_one'✝, map_mu …
    -/
    rfl
    /-
      🎉 no goals
    -/


