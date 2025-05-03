/-- The elements `1, x, ..., x ^ (d - 1)` for a basis for the `K`-module `K[x]`,
where `d` is the degree of the minimal polynomial of `x`. -/
noncomputable def adjoin.powerBasisAux {x : S} (hx : IsIntegral K x) :
    Basis (Fin (minpoly K x).natDegree) K (adjoin K ({x} : Set S)) := by
  /-
    K : Type u_1
    S : Type u_2
    inst✝² : Field K
    inst✝¹ : CommRing S
    inst✝ : Algebra K S
    x : S
    hx : IsIntegral K x
    ⊢ Basis (Fin (minpoly K x).natDegree) K (Subtype fun x_1 => Membership.mem (Al …
  -/
  have hST : Function.Injective (algebraMap (adjoin K ({x} : Set S)) S) := Subtype.coe_injective
  have hx' :
    IsIntegral K (⟨x, subset_adjoin (Set.mem_singleton x)⟩ : adjoin K ({x} : Set S)) := by
    apply (isIntegral_algebraMap_iff hST).mp
    convert hx
  apply
    @Basis.mk (Fin (minpoly K x).natDegree) _ (adjoin K {x}) fun i =>
      ⟨x, subset_adjoin (Set.mem_singleton x)⟩ ^ (i : ℕ)
  · have : LinearIndependent K _ := linearIndependent_pow
      (⟨x, self_mem_adjoin_singleton _ _⟩ : adjoin K {x})
    /-
      case hli
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      this : LinearIndependent K fun i => HPow.hPow ⟨x, ⋯⟩ ↑i
      ⊢ LinearIndependent K fun i => HPow.hPow ⟨x, ⋯⟩ ↑i
    -/
    rwa [← minpoly.algebraMap_eq hST] at this
    /-
      🎉 no goals
    -/
    /-
      case hsp
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      ⊢ LE.le Top.top (Submodule.span K (Set.range fun i => HPow.hPow ⟨x, ⋯⟩ ↑i))
    -/
  · rintro ⟨y, hy⟩ _
    /-
      case hsp.mk
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      y : S
      hy : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) y
      a✝ : Membership.mem Top.top ⟨y, hy⟩
      ⊢ Membership.mem (Submodule.span K (Set.range fun i => HPow.hPow ⟨x, ⋯⟩ ↑i)) ⟨ …
    -/
    have := hx'.mem_span_pow (y := ⟨y, hy⟩)
    /-
      case hsp.mk
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      y : S
      hy : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) y
      a✝ : Membership.mem Top.top ⟨y, hy⟩
      this : (Exists fun f => Eq ⟨y, hy⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)) → Membership …
      ⊢ Membership.mem (Submodule.span K (Set.range fun i => HPow.hPow ⟨x, ⋯⟩ ↑i)) ⟨ …
    -/
    rw [← minpoly.algebraMap_eq hST] at this
    /-
      case hsp.mk
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      y : S
      hy : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) y
      a✝ : Membership.mem Top.top ⟨y, hy⟩
      this : (Exists fun f => Eq ⟨y, hy⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)) → Membership …
      ⊢ Membership.mem (Submodule.span K (Set.range fun i => HPow.hPow ⟨x, ⋯⟩ ↑i)) ⟨ …
    -/
    apply this
    /-
      case hsp.mk
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      y : S
      hy : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) y
      a✝ : Membership.mem Top.top ⟨y, hy⟩
      this : (Exists fun f => Eq ⟨y, hy⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)) → Membership …
      ⊢ Exists fun f => Eq ⟨y, hy⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)
    -/
    rw [adjoin_singleton_eq_range_aeval] at hy
    /-
      case hsp.mk
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      y : S
      hy✝ : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) y
      hy : Membership.mem (Polynomial.aeval x).range y
      a✝ : Membership.mem Top.top ⟨y, hy✝⟩
      this : (Exists fun f => Eq ⟨y, hy✝⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)) → Membershi …
      ⊢ Exists fun f => Eq ⟨y, hy✝⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)
    -/
    obtain ⟨f, rfl⟩ := (aeval x).mem_range.mp hy
    /-
      case hsp.mk.intro
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      f : Polynomial K
      hy✝ : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) ((Polynomial.a …
      hy : Membership.mem (Polynomial.aeval x).range ((Polynomial.aeval x) f)
      a✝ : Membership.mem Top.top ⟨(Polynomial.aeval x) f, hy✝⟩
      this : (Exists fun f_1 => Eq ⟨(Polynomial.aeval x) f, hy✝⟩ ((Polynomial.aeval  …
      ⊢ Exists fun f_1 => Eq ⟨(Polynomial.aeval x) f, hy✝⟩ ((Polynomial.aeval ⟨x, ⋯⟩ …
    -/
    use f
    /-
      case h
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      f : Polynomial K
      hy✝ : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) ((Polynomial.a …
      hy : Membership.mem (Polynomial.aeval x).range ((Polynomial.aeval x) f)
      a✝ : Membership.mem Top.top ⟨(Polynomial.aeval x) f, hy✝⟩
      this : (Exists fun f_1 => Eq ⟨(Polynomial.aeval x) f, hy✝⟩ ((Polynomial.aeval  …
      ⊢ Eq ⟨(Polynomial.aeval x) f, hy✝⟩ ((Polynomial.aeval ⟨x, ⋯⟩) f)
    -/
    ext
    /-
      case h.a
      K : Type u_1
      S : Type u_2
      inst✝² : Field K
      inst✝¹ : CommRing S
      inst✝ : Algebra K S
      x : S
      hx : IsIntegral K x
      hST : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Alge …
      hx' : IsIntegral K ⟨x, ⋯⟩
      f : Polynomial K
      hy✝ : Membership.mem (Algebra.adjoin K (Singleton.singleton x)) ((Polynomial.a …
      hy : Membership.mem (Polynomial.aeval x).range ((Polynomial.aeval x) f)
      a✝ : Membership.mem Top.top ⟨(Polynomial.aeval x) f, hy✝⟩
      this : (Exists fun f_1 => Eq ⟨(Polynomial.aeval x) f, hy✝⟩ ((Polynomial.aeval  …
      ⊢ Eq ↑⟨(Polynomial.aeval x) f, hy✝⟩ ↑((Polynomial.aeval ⟨x, ⋯⟩) f)
    -/
    exact aeval_algebraMap_apply S (⟨x, _⟩ : adjoin K {x}) _
    /-
      🎉 no goals
    -/


/-- The power basis `1, x, ..., x ^ (d - 1)` for `K[x]`,
where `d` is the degree of the minimal polynomial of `x`. See `Algebra.adjoin.powerBasis'` for
a version over a more general base ring. -/
@[simps gen dim]
noncomputable def adjoin.powerBasis {x : S} (hx : IsIntegral K x) :
    PowerBasis K (adjoin K ({x} : Set S)) where
  gen := ⟨x, subset_adjoin (Set.mem_singleton x)⟩
  dim := (minpoly K x).natDegree
  basis := adjoin.powerBasisAux hx
                       /-
                         K : Type u_1
                         S : Type u_2
                         inst✝² : Field K
                         inst✝¹ : CommRing S
                         inst✝ : Algebra K S
                         x : S
                         hx : IsIntegral K x
                         i : Fin (minpoly K x).natDegree
                         ⊢ Eq ((Algebra.adjoin.powerBasisAux hx) i) (HPow.hPow ⟨x, ⋯⟩ ↑i)
                       -/
  basis_eq_pow i := by rw [adjoin.powerBasisAux, Basis.mk_apply]
                       /-
                         🎉 no goals
                       -/


/-- The power basis given by `x` if `B.gen ∈ adjoin K {x}`. See `PowerBasis.ofGenMemAdjoin'`
for a version over a more general base ring. -/
@[simps!]
noncomputable def PowerBasis.ofGenMemAdjoin {x : S} (B : PowerBasis K S) (hint : IsIntegral K x)
    (hx : B.gen ∈ adjoin K ({x} : Set S)) : PowerBasis K S :=
  (Algebra.adjoin.powerBasis hint).map <|
    (Subalgebra.equivOfEq _ _ <| PowerBasis.adjoin_eq_top_of_gen_mem_adjoin hx).trans
      Subalgebra.topEquiv


/-- If `B : PowerBasis S A` is such that `IsIntegral R B.gen`, then
`IsIntegral R (B.basis.repr (B.gen ^ n) i)` for all `i` if
`minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)`. This is the case if `R` is a GCD domain
and `S` is its fraction ring. -/
theorem repr_gen_pow_isIntegral (hB : IsIntegral R B.gen) [IsDomain S]
    (hmin : minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)) (n : ℕ) :
    ∀ i, IsIntegral R (B.basis.repr (B.gen ^ n) i) := by
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HPow.hPow B.gen n)) i)
  -/
  intro i
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    ⊢ IsIntegral R ((B.basis.repr (HPow.hPow B.gen n)) i)
  -/
  let Q := X ^ n %ₘ minpoly R B.gen
  have : B.gen ^ n = aeval B.gen Q := by
    rw [← @aeval_X_pow R _ _ _ _ B.gen, ← modByMonic_add_div (X ^ n) (minpoly.monic hB)]
    simp [Q]
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    ⊢ IsIntegral R ((B.basis.repr (HPow.hPow B.gen n)) i)
  -/
  by_cases hQ : Q = 0
    /-
      case pos
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      hB : IsIntegral R B.gen
      inst✝ : IsDomain S
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      i : Fin B.dim
      Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
      this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
      hQ : Eq Q 0
      ⊢ IsIntegral R ((B.basis.repr (HPow.hPow B.gen n)) i)
    -/
  · simp [this, hQ, isIntegral_zero]
    /-
      🎉 no goals
    -/
  have hlt : Q.natDegree < B.dim := by
    rw [← B.natDegree_minpoly, hmin, (minpoly.monic hB).natDegree_map,
      natDegree_lt_natDegree_iff hQ]
    letI : Nontrivial R := Nontrivial.of_polynomial_ne hQ
    exact degree_modByMonic_lt _ (minpoly.monic hB)
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    ⊢ IsIntegral R ((B.basis.repr (HPow.hPow B.gen n)) i)
  -/
  rw [this, aeval_eq_sum_range' hlt]
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    ⊢ IsIntegral R ((B.basis.repr ((Finset.range B.dim).sum fun i => HSMul.hSMul ( …
  -/
  simp only [map_sum, LinearEquiv.map_smulₛₗ, RingHom.id_apply, Finset.sum_apply']
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    ⊢ IsIntegral R ((Finset.range B.dim).sum fun k => (B.basis.repr (HSMul.hSMul ( …
  -/
  refine IsIntegral.sum _ fun j hj => ?_
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    j : Nat
    hj : Membership.mem (Finset.range B.dim) j
    ⊢ IsIntegral R ((B.basis.repr (HSMul.hSMul (Q.coeff j) (HPow.hPow B.gen j))) i)
  -/
  replace hj := Finset.mem_range.1 hj
  rw [← Fin.val_mk hj, ← B.basis_eq_pow, Algebra.smul_def, IsScalarTower.algebraMap_apply R S A, ←
    Algebra.smul_def, LinearEquiv.map_smul]
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    j : Nat
    hj : LT.lt j B.dim
    ⊢ IsIntegral R ((HSMul.hSMul ((algebraMap R S) (Q.coeff ↑⟨j, hj⟩)) (B.basis.re …
  -/
  simp only [algebraMap_smul, Finsupp.coe_smul, Pi.smul_apply, B.basis.repr_self_apply]
  /-
    case neg
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    i : Fin B.dim
    Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
    this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
    hQ : Not (Eq Q 0)
    hlt : LT.lt Q.natDegree B.dim
    j : Nat
    hj : LT.lt j B.dim
    ⊢ IsIntegral R (HSMul.hSMul (Q.coeff j) (ite (Eq ⟨j, hj⟩ i) 1 0))
  -/
  by_cases hij : (⟨j, hj⟩ : Fin _) = i
    /-
      case pos
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      hB : IsIntegral R B.gen
      inst✝ : IsDomain S
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      i : Fin B.dim
      Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
      this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
      hQ : Not (Eq Q 0)
      hlt : LT.lt Q.natDegree B.dim
      j : Nat
      hj : LT.lt j B.dim
      hij : Eq ⟨j, hj⟩ i
      ⊢ IsIntegral R (HSMul.hSMul (Q.coeff j) (ite (Eq ⟨j, hj⟩ i) 1 0))
    -/
  · simp only [hij, eq_self_iff_true, if_true]
    /-
      case pos
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      hB : IsIntegral R B.gen
      inst✝ : IsDomain S
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      i : Fin B.dim
      Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
      this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
      hQ : Not (Eq Q 0)
      hlt : LT.lt Q.natDegree B.dim
      j : Nat
      hj : LT.lt j B.dim
      hij : Eq ⟨j, hj⟩ i
      ⊢ IsIntegral R (HSMul.hSMul (Q.coeff j) 1)
    -/
    rw [Algebra.smul_def, mul_one]
    /-
      case pos
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      hB : IsIntegral R B.gen
      inst✝ : IsDomain S
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      i : Fin B.dim
      Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
      this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
      hQ : Not (Eq Q 0)
      hlt : LT.lt Q.natDegree B.dim
      j : Nat
      hj : LT.lt j B.dim
      hij : Eq ⟨j, hj⟩ i
      ⊢ IsIntegral R ((algebraMap R S) (Q.coeff j))
    -/
    exact isIntegral_algebraMap
    /-
      🎉 no goals
    -/
    /-
      case neg
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      hB : IsIntegral R B.gen
      inst✝ : IsDomain S
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      i : Fin B.dim
      Q : Polynomial R := (HPow.hPow Polynomial.X n).modByMonic (minpoly R B.gen)
      this : Eq (HPow.hPow B.gen n) ((Polynomial.aeval B.gen) Q)
      hQ : Not (Eq Q 0)
      hlt : LT.lt Q.natDegree B.dim
      j : Nat
      hj : LT.lt j B.dim
      hij : Not (Eq ⟨j, hj⟩ i)
      ⊢ IsIntegral R (HSMul.hSMul (Q.coeff j) (ite (Eq ⟨j, hj⟩ i) 1 0))
    -/
  · simp [hij, isIntegral_zero]
    /-
      🎉 no goals
    -/


/-- Let `B : PowerBasis S A` be such that `IsIntegral R B.gen`, and let `x y : A` be elements with
integral coordinates in the base `B.basis`. Then `IsIntegral R ((B.basis.repr (x * y) i)` for all
`i` if `minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)`. This is the case if `R` is a GCD
domain and `S` is its fraction ring. -/
theorem repr_mul_isIntegral (hB : IsIntegral R B.gen) [IsDomain S] {x y : A}
    (hx : ∀ i, IsIntegral R (B.basis.repr x i)) (hy : ∀ i, IsIntegral R (B.basis.repr y i))
    (hmin : minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)) :
    ∀ i, IsIntegral R (B.basis.repr (x * y) i) := by
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    x y : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hy : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr y) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HMul.hMul x y)) i)
  -/
  intro i
  rw [← B.basis.sum_repr x, ← B.basis.sum_repr y, Finset.sum_mul_sum, ← Finset.sum_product',
    map_sum, Finset.sum_apply']
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    x y : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hy : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr y) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    i : Fin B.dim
    ⊢ IsIntegral R ((SProd.sprod Finset.univ Finset.univ).sum fun k => (B.basis.re …
  -/
  refine IsIntegral.sum _ fun I _ => ?_
  simp only [Algebra.smul_mul_assoc, Algebra.mul_smul_comm, LinearEquiv.map_smulₛₗ,
    RingHom.id_apply, Finsupp.coe_smul, Pi.smul_apply, id.smul_eq_mul]
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    x y : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hy : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr y) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    i : Fin B.dim
    I : Prod (Fin B.dim) (Fin B.dim)
    x✝ : Membership.mem (SProd.sprod Finset.univ Finset.univ) I
    ⊢ IsIntegral R (HMul.hMul ((B.basis.repr y) I.2) (HMul.hMul ((B.basis.repr x)  …
  -/
  refine (hy _).mul ((hx _).mul ?_)
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    x y : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hy : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr y) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    i : Fin B.dim
    I : Prod (Fin B.dim) (Fin B.dim)
    x✝ : Membership.mem (SProd.sprod Finset.univ Finset.univ) I
    ⊢ IsIntegral R ((B.basis.repr (HMul.hMul (B.basis I.1) (B.basis I.2))) i)
  -/
  simp only [coe_basis, ← pow_add]
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    hB : IsIntegral R B.gen
    inst✝ : IsDomain S
    x y : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hy : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr y) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    i : Fin B.dim
    I : Prod (Fin B.dim) (Fin B.dim)
    x✝ : Membership.mem (SProd.sprod Finset.univ Finset.univ) I
    ⊢ IsIntegral R ((B.basis.repr (HPow.hPow B.gen (HAdd.hAdd ↑I.1 ↑I.2))) i)
  -/
  exact repr_gen_pow_isIntegral hB hmin _ _
  /-
    🎉 no goals
  -/


/-- Let `B : PowerBasis S A` be such that `IsIntegral R B.gen`, and let `x : A` be an element
with integral coordinates in the base `B.basis`. Then `IsIntegral R ((B.basis.repr (x ^ n) i)` for
all `i` and all `n` if `minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)`. This is the case
if `R` is a GCD domain and `S` is its fraction ring. -/
theorem repr_pow_isIntegral [IsDomain S] (hB : IsIntegral R B.gen) {x : A}
    (hx : ∀ i, IsIntegral R (B.basis.repr x i))
    (hmin : minpoly S B.gen = (minpoly R B.gen).map (algebraMap R S)) (n : ℕ) :
    ∀ i, IsIntegral R (B.basis.repr (x ^ n) i) := by
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    inst✝ : IsDomain S
    hB : IsIntegral R B.gen
    x : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HPow.hPow x n)) i)
  -/
  nontriviality A using Subsingleton.elim (x ^ n) 0, isIntegral_zero
  /-
    S : Type u_2
    inst✝⁷ : CommRing S
    R : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Algebra R S
    A : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    B : PowerBasis S A
    inst✝ : IsDomain S
    hB : IsIntegral R B.gen
    x : A
    hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
    hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
    n : Nat
    a✝ : Nontrivial A
    ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HPow.hPow x n)) i)
  -/
  revert hx
  refine Nat.case_strong_induction_on
    -- Porting note: had to hint what to induct on
    (p := fun n ↦ _ → ∀ (i : Fin B.dim), IsIntegral R (B.basis.repr (x ^ n) i))
    n ?_ fun n hn => ?_
    /-
      case refine_1
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      inst✝ : IsDomain S
      hB : IsIntegral R B.gen
      x : A
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      a✝ : Nontrivial A
      ⊢ (fun n => (∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)) → ∀ (i : Fi …
    -/
  · intro _ i
    rw [pow_zero, ← pow_zero B.gen, ← Fin.val_mk B.dim_pos, ← B.basis_eq_pow,
      B.basis.repr_self_apply]
    /-
      case refine_1
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      inst✝ : IsDomain S
      hB : IsIntegral R B.gen
      x : A
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n : Nat
      a✝¹ : Nontrivial A
      a✝ : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
      i : Fin B.dim
      ⊢ IsIntegral R (ite (Eq ⟨0, ⋯⟩ i) 1 0)
    -/
    split_ifs
      /-
        case pos
        S : Type u_2
        inst✝⁷ : CommRing S
        R : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : Algebra R S
        A : Type u_4
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Algebra S A
        inst✝¹ : IsScalarTower R S A
        B : PowerBasis S A
        inst✝ : IsDomain S
        hB : IsIntegral R B.gen
        x : A
        hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
        n : Nat
        a✝¹ : Nontrivial A
        a✝ : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
        i : Fin B.dim
        h✝ : Eq ⟨0, ⋯⟩ i
        ⊢ IsIntegral R 1
      -/
    · exact isIntegral_one
      /-
        🎉 no goals
      -/
      /-
        case neg
        S : Type u_2
        inst✝⁷ : CommRing S
        R : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : Algebra R S
        A : Type u_4
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Algebra S A
        inst✝¹ : IsScalarTower R S A
        B : PowerBasis S A
        inst✝ : IsDomain S
        hB : IsIntegral R B.gen
        x : A
        hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
        n : Nat
        a✝¹ : Nontrivial A
        a✝ : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
        i : Fin B.dim
        h✝ : Not (Eq ⟨0, ⋯⟩ i)
        ⊢ IsIntegral R 0
      -/
    · exact isIntegral_zero
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      inst✝ : IsDomain S
      hB : IsIntegral R B.gen
      x : A
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n✝ : Nat
      a✝ : Nontrivial A
      n : Nat
      hn : ∀ (m : Nat), LE.le m n → (fun n => (∀ (i : Fin B.dim), IsIntegral R ((B.b …
      ⊢ (fun n => (∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)) → ∀ (i : Fi …
    -/
  · intro hx
    /-
      case refine_2
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      inst✝ : IsDomain S
      hB : IsIntegral R B.gen
      x : A
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n✝ : Nat
      a✝ : Nontrivial A
      n : Nat
      hn : ∀ (m : Nat), LE.le m n → (fun n => (∀ (i : Fin B.dim), IsIntegral R ((B.b …
      hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
      ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HPow.hPow x (HAdd.hAdd n 1)) …
    -/
    rw [pow_succ]
    /-
      case refine_2
      S : Type u_2
      inst✝⁷ : CommRing S
      R : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : Algebra R S
      A : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      B : PowerBasis S A
      inst✝ : IsDomain S
      hB : IsIntegral R B.gen
      x : A
      hmin : Eq (minpoly S B.gen) (Polynomial.map (algebraMap R S) (minpoly R B.gen))
      n✝ : Nat
      a✝ : Nontrivial A
      n : Nat
      hn : ∀ (m : Nat), LE.le m n → (fun n => (∀ (i : Fin B.dim), IsIntegral R ((B.b …
      hx : ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr x) i)
      ⊢ ∀ (i : Fin B.dim), IsIntegral R ((B.basis.repr (HMul.hMul (HPow.hPow x n) x) …
    -/
    exact repr_mul_isIntegral hB (fun _ => hn _ le_rfl (fun _ => hx _) _) hx hmin
    /-
      🎉 no goals
    -/


/-- Let `B B' : PowerBasis K S` be such that `IsIntegral R B.gen`, and let `P : R[X]` be such that
`aeval B.gen P = B'.gen`. Then `IsIntegral R (B.basis.to_matrix B'.basis i j)` for all `i` and `j`
if `minpoly K B.gen = (minpoly R B.gen).map (algebraMap R L)`. This is the case
if `R` is a GCD domain and `K` is its fraction ring. -/
theorem toMatrix_isIntegral {B B' : PowerBasis K S} {P : R[X]} (h : aeval B.gen P = B'.gen)
    (hB : IsIntegral R B.gen) (hmin : minpoly K B.gen = (minpoly R B.gen).map (algebraMap R K)) :
    ∀ i j, IsIntegral R (B.basis.toMatrix B'.basis i j) := by
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    ⊢ ∀ (i : Fin B.dim) (j : Fin B'.dim), IsIntegral R (B.basis.toMatrix (⇑B'.basi …
  -/
  intro i j
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    i : Fin B.dim
    j : Fin B'.dim
    ⊢ IsIntegral R (B.basis.toMatrix (⇑B'.basis) i j)
  -/
  rw [B.basis.toMatrix_apply, B'.coe_basis]
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    i : Fin B.dim
    j : Fin B'.dim
    ⊢ IsIntegral R ((B.basis.repr ((fun i => HPow.hPow B'.gen ↑i) j)) i)
  -/
  refine repr_pow_isIntegral hB (fun i => ?_) hmin _ _
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    i✝ : Fin B.dim
    j : Fin B'.dim
    i : Fin B.dim
    ⊢ IsIntegral R ((B.basis.repr B'.gen) i)
  -/
  rw [← h, aeval_eq_sum_range, map_sum, Finset.sum_apply']
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    i✝ : Fin B.dim
    j : Fin B'.dim
    i : Fin B.dim
    ⊢ IsIntegral R ((Finset.range (HAdd.hAdd P.natDegree 1)).sum fun k => (B.basis …
  -/
  refine IsIntegral.sum _ fun n _ => ?_
  rw [Algebra.smul_def, IsScalarTower.algebraMap_apply R K S, ← Algebra.smul_def,
    LinearEquiv.map_smul, algebraMap_smul]
  /-
    K : Type u_1
    S : Type u_2
    inst✝⁶ : Field K
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra K S
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : Algebra R S
    inst✝¹ : Algebra R K
    inst✝ : IsScalarTower R K S
    B B' : PowerBasis K S
    P : Polynomial R
    h : Eq ((Polynomial.aeval B.gen) P) B'.gen
    hB : IsIntegral R B.gen
    hmin : Eq (minpoly K B.gen) (Polynomial.map (algebraMap R K) (minpoly R B.gen))
    i✝ : Fin B.dim
    j : Fin B'.dim
    i : Fin B.dim
    n : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd P.natDegree 1)) n
    ⊢ IsIntegral R ((HSMul.hSMul (P.coeff n) (B.basis.repr (HPow.hPow B.gen n))) i)
  -/
  exact (repr_gen_pow_isIntegral hB hmin _ _).smul _
  /-
    🎉 no goals
  -/


