/-- The rank of a module, defined as a term of type `Cardinal`.

We define this as the supremum of the cardinalities of linearly independent subsets.

For a free module over any ring satisfying the strong rank condition
(e.g. left-noetherian rings, commutative rings, and in particular division rings and fields),
this is the same as the dimension of the space (i.e. the cardinality of any basis).

In particular this agrees with the usual notion of the dimension of a vector space. -/
@[stacks 09G3 "first part"]
protected irreducible_def Module.rank : Cardinal :=
  ⨆ ι : { s : Set M // LinearIndependent R ((↑) : s → M) }, (#ι.1)


theorem rank_le_card : Module.rank R M ≤ #M :=
  (Module.rank_def _ _).trans_le (ciSup_le' fun _ ↦ mk_set_le _)


lemma nonempty_linearIndependent_set : Nonempty {s : Set M // LinearIndependent R ((↑) : s → M)} :=
  ⟨⟨∅, linearIndependent_empty _ _⟩⟩


theorem cardinal_lift_le_rank {ι : Type w} {v : ι → M}
    (hv : LinearIndependent R v) :
    Cardinal.lift.{v} #ι ≤ Cardinal.lift.{w} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    ι : Type w
    v : ι → M
    hv : LinearIndependent R v
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} (Module.r …
  -/
  rw [Module.rank]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    ι : Type w
    v : ι → M
    hv : LinearIndependent R v
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} (iSup fun …
  -/
  refine le_trans ?_ (lift_le.mpr <| le_ciSup (bddAbove_range _) ⟨_, hv.coe_range⟩)
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    ι : Type w
    v : ι → M
    hv : LinearIndependent R v
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk ι)) (Cardinal.lift.{w, v} (Cardinal …
  -/
  exact lift_mk_le'.mpr ⟨(Equiv.ofInjective _ hv.injective).toEmbedding⟩
  /-
    🎉 no goals
  -/


lemma aleph0_le_rank {ι : Type w} [Infinite ι] {v : ι → M}
    (hv : LinearIndependent R v) : ℵ₀ ≤ Module.rank R M :=
  aleph0_le_lift.mp <| (aleph0_le_lift.mpr <| aleph0_le_mk ι).trans hv.cardinal_lift_le_rank


theorem cardinal_le_rank {ι : Type v} {v : ι → M}
    (hv : LinearIndependent R v) : #ι ≤ Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    ι : Type v
    v : ι → M
    hv : LinearIndependent R v
    ⊢ LE.le (Cardinal.mk ι) (Module.rank R M)
  -/
  simpa using hv.cardinal_lift_le_rank
  /-
    🎉 no goals
  -/


theorem cardinal_le_rank' {s : Set M}
    (hs : LinearIndependent R (fun x => x : s → M)) : #s ≤ Module.rank R M :=
  hs.cardinal_le_rank


/-- If `M / R` and `M' / R'` are modules, `i : R' → R` is a map which sends non-zero elements to
non-zero elements, `j : M →+ M'` is an injective group homomorphism, such that the scalar
multiplications on `M` and `M'` are compatible, then the rank of `M / R` is smaller than or equal to
the rank of `M' / R'`. As a special case, taking `R = R'` it is
`LinearMap.lift_rank_le_of_injective`. -/
theorem lift_rank_le_of_injective_injective (i : R' → R) (j : M →+ M')
    (hi : ∀ r, i r = 0 → r = 0) (hj : Injective j)
    (hc : ∀ (r : R') (m : M), j (i r • m) = r • j m) :
    lift.{v'} (Module.rank R M) ≤ lift.{v} (Module.rank R' M') := by
  /-
    R : Type u
    R' : Type u'
    M : Type v
    M' : Type v'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    i : R' → R
    j : AddMonoidHom M M'
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : Function.Injective ⇑j
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    ⊢ LE.le (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Modu …
  -/
  simp_rw [Module.rank, lift_iSup (bddAbove_range _)]
  exact ciSup_mono' (bddAbove_range _) fun ⟨s, h⟩ ↦ ⟨⟨j '' s,
    (h.map_of_injective_injective i j hi (fun _ _ ↦ hj <| by rwa [j.map_zero]) hc).image⟩,
      lift_mk_le'.mpr ⟨(Equiv.Set.image j s hj).toEmbedding⟩⟩


/-- If `M / R` and `M' / R'` are modules, `i : R → R'` is a surjective map which maps zero to zero,
`j : M →+ M'` is an injective group homomorphism, such that the scalar multiplications on `M` and
`M'` are compatible, then the rank of `M / R` is smaller than or equal to the rank of `M' / R'`.
As a special case, taking `R = R'` it is `LinearMap.lift_rank_le_of_injective`. -/
theorem lift_rank_le_of_surjective_injective (i : ZeroHom R R') (j : M →+ M')
    (hi : Surjective i) (hj : Injective j) (hc : ∀ (r : R) (m : M), j (r • m) = i r • j m) :
    lift.{v'} (Module.rank R M) ≤ lift.{v} (Module.rank R' M') := by
  /-
    R : Type u
    R' : Type u'
    M : Type v
    M' : Type v'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    ⊢ LE.le (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Modu …
  -/
  obtain ⟨i', hi'⟩ := hi.hasRightInverse
  /-
    case intro
    R : Type u
    R' : Type u'
    M : Type v
    M' : Type v'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    i' : R' → R
    hi' : Function.RightInverse i' ⇑i
    ⊢ LE.le (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Modu …
  -/
  refine lift_rank_le_of_injective_injective i' j (fun _ h ↦ ?_) hj fun r m ↦ ?_
    /-
      case intro.refine_1
      R : Type u
      R' : Type u'
      M : Type v
      M' : Type v'
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : Ring R'
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R' M'
      i : ZeroHom R R'
      j : AddMonoidHom M M'
      hi : Function.Surjective ⇑i
      hj : Function.Injective ⇑j
      hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
      i' : R' → R
      hi' : Function.RightInverse i' ⇑i
      x✝ : R'
      h : Eq (i' x✝) 0
      ⊢ Eq x✝ 0
    -/
  · apply_fun i at h
    /-
      case intro.refine_1
      R : Type u
      R' : Type u'
      M : Type v
      M' : Type v'
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : Ring R'
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R' M'
      i : ZeroHom R R'
      j : AddMonoidHom M M'
      hi : Function.Surjective ⇑i
      hj : Function.Injective ⇑j
      hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
      i' : R' → R
      hi' : Function.RightInverse i' ⇑i
      x✝ : R'
      h : Eq (i (i' x✝)) (i 0)
      ⊢ Eq x✝ 0
    -/
    rwa [hi', i.map_zero] at h
    /-
      🎉 no goals
    -/
  /-
    case intro.refine_2
    R : Type u
    R' : Type u'
    M : Type v
    M' : Type v'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    i' : R' → R
    hi' : Function.RightInverse i' ⇑i
    r : R'
    m : M
    ⊢ Eq (j (HSMul.hSMul (i' r) m)) (HSMul.hSMul r (j m))
  -/
  rw [hc (i' r) m, hi']
  /-
    🎉 no goals
  -/


/-- If `M / R` and `M' / R'` are modules, `i : R → R'` is a bijective map which maps zero to zero,
`j : M ≃+ M'` is a group isomorphism, such that the scalar multiplications on `M` and `M'` are
compatible, then the rank of `M / R` is equal to the rank of `M' / R'`.
As a special case, taking `R = R'` it is `LinearEquiv.lift_rank_eq`. -/
theorem lift_rank_eq_of_equiv_equiv (i : ZeroHom R R') (j : M ≃+ M')
    (hi : Bijective i) (hc : ∀ (r : R) (m : M), j (r • m) = i r • j m) :
    lift.{v'} (Module.rank R M) = lift.{v} (Module.rank R' M') :=
  (lift_rank_le_of_surjective_injective i j hi.2 j.injective hc).antisymm <|
                                                                       /-
                                                                         R : Type u
                                                                         R' : Type u'
                                                                         M : Type v
                                                                         M' : Type v'
                                                                         inst✝⁵ : Ring R
                                                                         inst✝⁴ : AddCommGroup M
                                                                         inst✝³ : Module R M
                                                                         inst✝² : Ring R'
                                                                         inst✝¹ : AddCommGroup M'
                                                                         inst✝ : Module R' M'
                                                                         i : ZeroHom R R'
                                                                         j : AddEquiv M M'
                                                                         hi : Function.Bijective ⇑i
                                                                         hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
                                                                         x✝¹ : R
                                                                         x✝ : Eq (i x✝¹) 0
                                                                         ⊢ Eq (i x✝¹) (i 0)
                                                                       -/
    lift_rank_le_of_injective_injective i j.symm (fun _ _ ↦ hi.1 <| by rwa [i.map_zero])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                         /-
                                                           R : Type u
                                                           R' : Type u'
                                                           M : Type v
                                                           M' : Type v'
                                                           inst✝⁵ : Ring R
                                                           inst✝⁴ : AddCommGroup M
                                                           inst✝³ : Module R M
                                                           inst✝² : Ring R'
                                                           inst✝¹ : AddCommGroup M'
                                                           inst✝ : Module R' M'
                                                           i : ZeroHom R R'
                                                           j : AddEquiv M M'
                                                           hi : Function.Bijective ⇑i
                                                           hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
                                                           x✝¹ : R
                                                           x✝ : M'
                                                           ⊢ Eq (HSMul.hSMul (i x✝¹) x✝) (j (HSMul.hSMul x✝¹ (↑j.symm x✝)))
                                                         -/
      j.symm.injective fun _ _ ↦ j.symm_apply_eq.2 <| by erw [hc, j.apply_symm_apply]
                                                         /-
                                                           🎉 no goals
                                                         -/

/-- The same-universe version of `lift_rank_le_of_injective_injective`. -/
theorem rank_le_of_injective_injective (i : R' → R) (j : M →+ M₁)
    (hi : ∀ r, i r = 0 → r = 0) (hj : Injective j)
    (hc : ∀ (r : R') (m : M), j (i r • m) = r • j m) :
    Module.rank R M ≤ Module.rank R' M₁ := by
  /-
    R : Type u
    R' : Type u'
    M M₁ : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R' M₁
    i : R' → R
    j : AddMonoidHom M M₁
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : Function.Injective ⇑j
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    ⊢ LE.le (Module.rank R M) (Module.rank R' M₁)
  -/
  simpa only [lift_id] using lift_rank_le_of_injective_injective i j hi hj hc
  /-
    🎉 no goals
  -/


/-- The same-universe version of `lift_rank_le_of_surjective_injective`. -/
theorem rank_le_of_surjective_injective (i : ZeroHom R R') (j : M →+ M₁)
    (hi : Surjective i) (hj : Injective j)
    (hc : ∀ (r : R) (m : M), j (r • m) = i r • j m) :
    Module.rank R M ≤ Module.rank R' M₁ := by
  /-
    R : Type u
    R' : Type u'
    M M₁ : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R' M₁
    i : ZeroHom R R'
    j : AddMonoidHom M M₁
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    ⊢ LE.le (Module.rank R M) (Module.rank R' M₁)
  -/
  simpa only [lift_id] using lift_rank_le_of_surjective_injective i j hi hj hc
  /-
    🎉 no goals
  -/


/-- The same-universe version of `lift_rank_eq_of_equiv_equiv`. -/
theorem rank_eq_of_equiv_equiv (i : ZeroHom R R') (j : M ≃+ M₁)
    (hi : Bijective i) (hc : ∀ (r : R) (m : M), j (r • m) = i r • j m) :
    Module.rank R M = Module.rank R' M₁ := by
  /-
    R : Type u
    R' : Type u'
    M M₁ : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R' M₁
    i : ZeroHom R R'
    j : AddEquiv M M₁
    hi : Function.Bijective ⇑i
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    ⊢ Eq (Module.rank R M) (Module.rank R' M₁)
  -/
  simpa only [lift_id] using lift_rank_eq_of_equiv_equiv i j hi hc
  /-
    🎉 no goals
  -/


/-- If `S / R` and `S' / R'` are algebras, `i : R' →+* R` and `j : S →+* S'` are injective ring
homomorphisms, such that `R' → R → S → S'` and `R' → S'` commute, then the rank of `S / R` is
smaller than or equal to the rank of `S' / R'`. -/
theorem lift_rank_le_of_injective_injective
    (i : R' →+* R) (j : S →+* S') (hi : Injective i) (hj : Injective j)
    (hc : (j.comp (algebraMap R S)).comp i = algebraMap R' S') :
    lift.{v'} (Module.rank R S) ≤ lift.{v} (Module.rank R' S') := by
  refine _root_.lift_rank_le_of_injective_injective i j
    (fun _ _ ↦ hi <| by rwa [i.map_zero]) hj fun r _ ↦ ?_
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R' R
    j : RingHom S S'
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((j.comp (algebraMap R S)).comp i) (algebraMap R' S')
    r : R'
    x✝ : S
    ⊢ Eq (↑j (HSMul.hSMul (i r) x✝)) (HSMul.hSMul r (↑j x✝))
  -/
  have := congr($hc r)
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R' R
    j : RingHom S S'
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((j.comp (algebraMap R S)).comp i) (algebraMap R' S')
    r : R'
    x✝ : S
    this : Eq (((j.comp (algebraMap R S)).comp i) r) ((algebraMap R' S') r)
    ⊢ Eq (↑j (HSMul.hSMul (i r) x✝)) (HSMul.hSMul r (↑j x✝))
  -/
  simp only [RingHom.coe_comp, comp_apply] at this
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R' R
    j : RingHom S S'
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((j.comp (algebraMap R S)).comp i) (algebraMap R' S')
    r : R'
    x✝ : S
    this : Eq (j ((algebraMap R S) (i r))) ((algebraMap R' S') r)
    ⊢ Eq (↑j (HSMul.hSMul (i r) x✝)) (HSMul.hSMul r (↑j x✝))
  -/
  simp_rw [smul_def, AddMonoidHom.coe_coe, map_mul, this]
  /-
    🎉 no goals
  -/


/-- If `S / R` and `S' / R'` are algebras, `i : R →+* R'` is a surjective ring homomorphism,
`j : S →+* S'` is an injective ring homomorphism, such that `R → R' → S'` and `R → S → S'` commute,
then the rank of `S / R` is smaller than or equal to the rank of `S' / R'`. -/
theorem lift_rank_le_of_surjective_injective
    (i : R →+* R') (j : S →+* S') (hi : Surjective i) (hj : Injective j)
    (hc : (algebraMap R' S').comp i = j.comp (algebraMap R S)) :
    lift.{v'} (Module.rank R S) ≤ lift.{v} (Module.rank R' S') := by
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R R'
    j : RingHom S S'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((algebraMap R' S').comp i) (j.comp (algebraMap R S))
    ⊢ LE.le (Cardinal.lift.{v', v} (Module.rank R S)) (Cardinal.lift.{v, v'} (Modu …
  -/
  refine _root_.lift_rank_le_of_surjective_injective i j hi hj fun r _ ↦ ?_
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R R'
    j : RingHom S S'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((algebraMap R' S').comp i) (j.comp (algebraMap R S))
    r : R
    x✝ : S
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  have := congr($hc r)
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R R'
    j : RingHom S S'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((algebraMap R' S').comp i) (j.comp (algebraMap R S))
    r : R
    x✝ : S
    this : Eq (((algebraMap R' S').comp i) r) ((j.comp (algebraMap R S)) r)
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  simp only [RingHom.coe_comp, comp_apply] at this
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R R'
    j : RingHom S S'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((algebraMap R' S').comp i) (j.comp (algebraMap R S))
    r : R
    x✝ : S
    this : Eq ((algebraMap R' S') (i r)) (j ((algebraMap R S) r))
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  simp only [smul_def, AddMonoidHom.coe_coe, map_mul, ZeroHom.coe_coe, this]
  /-
    🎉 no goals
  -/


/-- If `S / R` and `S' / R'` are algebras, `i : R ≃+* R'` and `j : S ≃+* S'` are
ring isomorphisms, such that `R → R' → S'` and `R → S → S'` commute,
then the rank of `S / R` is equal to the rank of `S' / R'`. -/
theorem lift_rank_eq_of_equiv_equiv (i : R ≃+* R') (j : S ≃+* S')
    (hc : (algebraMap R' S').comp i.toRingHom = j.toRingHom.comp (algebraMap R S)) :
    lift.{v'} (Module.rank R S) = lift.{v} (Module.rank R' S') := by
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingEquiv R R'
    j : RingEquiv S S'
    hc : Eq ((algebraMap R' S').comp i.toRingHom) (j.toRingHom.comp (algebraMap R  …
    ⊢ Eq (Cardinal.lift.{v', v} (Module.rank R S)) (Cardinal.lift.{v, v'} (Module. …
  -/
  refine _root_.lift_rank_eq_of_equiv_equiv i j i.bijective fun r _ ↦ ?_
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingEquiv R R'
    j : RingEquiv S S'
    hc : Eq ((algebraMap R' S').comp i.toRingHom) (j.toRingHom.comp (algebraMap R  …
    r : R
    x✝ : S
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  have := congr($hc r)
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingEquiv R R'
    j : RingEquiv S S'
    hc : Eq ((algebraMap R' S').comp i.toRingHom) (j.toRingHom.comp (algebraMap R  …
    r : R
    x✝ : S
    this : Eq (((algebraMap R' S').comp i.toRingHom) r) ((j.toRingHom.comp (algebr …
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  simp only [RingEquiv.toRingHom_eq_coe, RingHom.coe_comp, RingHom.coe_coe, comp_apply] at this
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    S' : Type v'
    inst✝² : CommRing R'
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingEquiv R R'
    j : RingEquiv S S'
    hc : Eq ((algebraMap R' S').comp i.toRingHom) (j.toRingHom.comp (algebraMap R  …
    r : R
    x✝ : S
    this : Eq ((algebraMap R' S') (i r)) (j ((algebraMap R S) r))
    ⊢ Eq (↑j (HSMul.hSMul r x✝)) (HSMul.hSMul (↑i r) (↑j x✝))
  -/
  simp only [smul_def, RingEquiv.coe_toAddEquiv, map_mul, ZeroHom.coe_coe, this]
  /-
    🎉 no goals
  -/


/-- The same-universe version of `Algebra.lift_rank_le_of_injective_injective`. -/
theorem rank_le_of_injective_injective
    (i : R' →+* R) (j : S →+* S') (hi : Injective i) (hj : Injective j)
    (hc : (j.comp (algebraMap R S)).comp i = algebraMap R' S') :
    Module.rank R S ≤ Module.rank R' S' := by
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    inst✝² : CommRing R'
    S' : Type v
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R' R
    j : RingHom S S'
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((j.comp (algebraMap R S)).comp i) (algebraMap R' S')
    ⊢ LE.le (Module.rank R S) (Module.rank R' S')
  -/
  simpa only [lift_id] using lift_rank_le_of_injective_injective i j hi hj hc
  /-
    🎉 no goals
  -/


/-- The same-universe version of `Algebra.lift_rank_le_of_surjective_injective`. -/
theorem rank_le_of_surjective_injective
    (i : R →+* R') (j : S →+* S') (hi : Surjective i) (hj : Injective j)
    (hc : (algebraMap R' S').comp i = j.comp (algebraMap R S)) :
    Module.rank R S ≤ Module.rank R' S' := by
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    inst✝² : CommRing R'
    S' : Type v
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingHom R R'
    j : RingHom S S'
    hi : Function.Surjective ⇑i
    hj : Function.Injective ⇑j
    hc : Eq ((algebraMap R' S').comp i) (j.comp (algebraMap R S))
    ⊢ LE.le (Module.rank R S) (Module.rank R' S')
  -/
  simpa only [lift_id] using lift_rank_le_of_surjective_injective i j hi hj hc
  /-
    🎉 no goals
  -/


/-- The same-universe version of `Algebra.lift_rank_eq_of_equiv_equiv`. -/
theorem rank_eq_of_equiv_equiv (i : R ≃+* R') (j : S ≃+* S')
    (hc : (algebraMap R' S').comp i.toRingHom = j.toRingHom.comp (algebraMap R S)) :
    Module.rank R S = Module.rank R' S' := by
  /-
    R : Type w
    S : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    R' : Type w'
    inst✝² : CommRing R'
    S' : Type v
    inst✝¹ : Ring S'
    inst✝ : Algebra R' S'
    i : RingEquiv R R'
    j : RingEquiv S S'
    hc : Eq ((algebraMap R' S').comp i.toRingHom) (j.toRingHom.comp (algebraMap R  …
    ⊢ Eq (Module.rank R S) (Module.rank R' S')
  -/
  simpa only [lift_id] using lift_rank_eq_of_equiv_equiv i j hc
  /-
    🎉 no goals
  -/


theorem LinearMap.lift_rank_le_of_injective (f : M →ₗ[R] M') (i : Injective f) :
    Cardinal.lift.{v'} (Module.rank R M) ≤ Cardinal.lift.{v} (Module.rank R M') :=
  lift_rank_le_of_injective_injective (RingHom.id R) f (fun _ h ↦ h) i f.map_smul


theorem LinearMap.rank_le_of_injective (f : M →ₗ[R] M₁) (i : Injective f) :
    Module.rank R M ≤ Module.rank R M₁ :=
  Cardinal.lift_le.1 (f.lift_rank_le_of_injective i)


/-- The rank of the range of a linear map is at most the rank of the source. -/
-- The proof is: a free submodule of the range lifts to a free submodule of the
-- source, by arbitrarily lifting a basis.
theorem lift_rank_range_le (f : M →ₗ[R] M') : Cardinal.lift.{v}
    (Module.rank R (LinearMap.range f)) ≤ Cardinal.lift.{v'} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R (Subtype fun x => Membership.mem …
  -/
  simp only [Module.rank_def]
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ LE.le (Cardinal.lift.{v, v'} (iSup fun ι => Cardinal.mk ↑↑ι)) (Cardinal.lift …
  -/
  rw [Cardinal.lift_iSup (Cardinal.bddAbove_range _)]
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ LE.le (iSup fun i => Cardinal.lift.{v, v'} (Cardinal.mk ↑↑i)) (Cardinal.lift …
  -/
  apply ciSup_le'
  /-
    case h
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ ∀ (i : Subtype fun s => LinearIndependent R Subtype.val), LE.le (Cardinal.li …
  -/
  rintro ⟨s, li⟩
  /-
    case h.mk
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
    li : LinearIndependent R Subtype.val
    ⊢ LE.le (Cardinal.lift.{v, v'} (Cardinal.mk ↑↑⟨s, li⟩)) (Cardinal.lift.{v', v} …
  -/
  apply le_trans
  /-
    case h.mk.a
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
    li : LinearIndependent R Subtype.val
    ⊢ LE.le (Cardinal.lift.{v, v'} (Cardinal.mk ↑↑⟨s, li⟩)) ?h.mk.b
  -/
  swap
    /-
      case h.mk.a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      li : LinearIndependent R Subtype.val
      ⊢ LE.le ?h.mk.b (Cardinal.lift.{v', v} (iSup fun ι => Cardinal.mk ↑↑ι))
    -/
  · apply Cardinal.lift_le.mpr
    /-
      case h.mk.a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      li : LinearIndependent R Subtype.val
      ⊢ LE.le ?m.107695 (iSup fun ι => Cardinal.mk ↑↑ι)
    -/
    refine le_ciSup (Cardinal.bddAbove_range _) ⟨rangeSplitting f '' s, ?_⟩
    /-
      case h.mk.a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      li : LinearIndependent R Subtype.val
      ⊢ LinearIndependent R Subtype.val
    -/
    apply LinearIndependent.of_comp f.rangeRestrict
    /-
      case h.mk.a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      li : LinearIndependent R Subtype.val
      ⊢ LinearIndependent R (Function.comp (⇑f.rangeRestrict) Subtype.val)
    -/
    convert li.comp (Equiv.Set.rangeSplittingImageEquiv f s) (Equiv.injective _) using 1
    /-
      🎉 no goals
    -/
    /-
      case h.mk.a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      li : LinearIndependent R Subtype.val
      ⊢ LE.le (Cardinal.lift.{v, v'} (Cardinal.mk ↑↑⟨s, li⟩)) (Cardinal.lift.{v', v} …
    -/
  · exact (Cardinal.lift_mk_eq'.mpr ⟨Equiv.Set.rangeSplittingImageEquiv f s⟩).ge
    /-
      🎉 no goals
    -/


theorem rank_range_le (f : M →ₗ[R] M₁) : Module.rank R (LinearMap.range f) ≤ Module.rank R M := by
  /-
    R : Type u
    M M₁ : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    f : LinearMap (RingHom.id R) M M₁
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (LinearMap.range f) x) …
  -/
  simpa using lift_rank_range_le f
  /-
    🎉 no goals
  -/


theorem lift_rank_map_le (f : M →ₗ[R] M') (p : Submodule R M) :
    Cardinal.lift.{v} (Module.rank R (p.map f)) ≤ Cardinal.lift.{v'} (Module.rank R p) := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    p : Submodule R M
    ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R (Subtype fun x => Membership.mem …
  -/
  have h := lift_rank_range_le (f.comp (Submodule.subtype p))
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    p : Submodule R M
    h : LE.le (Cardinal.lift.{v, v'} (Module.rank R (Subtype fun x => Membership.m …
    ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R (Subtype fun x => Membership.mem …
  -/
  rwa [LinearMap.range_comp, range_subtype] at h
  /-
    🎉 no goals
  -/


theorem rank_map_le (f : M →ₗ[R] M₁) (p : Submodule R M) :
                                                    /-
                                                      R : Type u
                                                      M M₁ : Type v
                                                      inst✝⁴ : Ring R
                                                      inst✝³ : AddCommGroup M
                                                      inst✝² : Module R M
                                                      inst✝¹ : AddCommGroup M₁
                                                      inst✝ : Module R M₁
                                                      f : LinearMap (RingHom.id R) M M₁
                                                      p : Submodule R M
                                                      ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Submodule.map f p) x) …
                                                    -/
    Module.rank R (p.map f) ≤ Module.rank R p := by simpa using lift_rank_map_le f p
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma Submodule.rank_mono {s t : Submodule R M} (h : s ≤ t) : Module.rank R s ≤ Module.rank R t :=
  (Submodule.inclusion h).rank_le_of_injective fun ⟨x, _⟩ ⟨y, _⟩ eq =>
    Subtype.eq <| show x = y from Subtype.ext_iff_val.1 eq


@[deprecated (since := "2024-09-30")] alias rank_le_of_submodule := Submodule.rank_mono


/-- Two linearly equivalent vector spaces have the same dimension, a version with different
universes. -/
theorem LinearEquiv.lift_rank_eq (f : M ≃ₗ[R] M') :
    Cardinal.lift.{v'} (Module.rank R M) = Cardinal.lift.{v} (Module.rank R M') := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearEquiv (RingHom.id R) M M'
    ⊢ Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Module. …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearEquiv (RingHom.id R) M M'
      ⊢ LE.le (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Modu …
    -/
  · exact f.toLinearMap.lift_rank_le_of_injective f.injective
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearEquiv (RingHom.id R) M M'
      ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R M')) (Cardinal.lift.{v', v} (Mod …
    -/
  · exact f.symm.toLinearMap.lift_rank_le_of_injective f.symm.injective
    /-
      🎉 no goals
    -/


/-- Two linearly equivalent vector spaces have the same dimension. -/
theorem LinearEquiv.rank_eq (f : M ≃ₗ[R] M₁) : Module.rank R M = Module.rank R M₁ :=
  Cardinal.lift_inj.1 f.lift_rank_eq


theorem lift_rank_range_of_injective (f : M →ₗ[R] M') (h : Injective f) :
    lift.{v} (Module.rank R (LinearMap.range f)) = lift.{v'} (Module.rank R M) :=
  (LinearEquiv.ofInjective f h).lift_rank_eq.symm


theorem rank_range_of_injective (f : M →ₗ[R] M₁) (h : Injective f) :
    Module.rank R (LinearMap.range f) = Module.rank R M :=
  (LinearEquiv.ofInjective f h).rank_eq.symm


theorem LinearEquiv.lift_rank_map_eq (f : M ≃ₗ[R] M') (p : Submodule R M) :
    lift.{v} (Module.rank R (p.map (f : M →ₗ[R] M'))) = lift.{v'} (Module.rank R p) :=
  (f.submoduleMap p).lift_rank_eq.symm


/-- Pushforwards of submodules along a `LinearEquiv` have the same dimension. -/
theorem LinearEquiv.rank_map_eq (f : M ≃ₗ[R] M₁) (p : Submodule R M) :
    Module.rank R (p.map (f : M →ₗ[R] M₁)) = Module.rank R p :=
  (f.submoduleMap p).rank_eq.symm


@[simp]
theorem rank_top : Module.rank R (⊤ : Submodule R M) = Module.rank R M :=
  (LinearEquiv.ofTop ⊤ rfl).rank_eq


theorem rank_range_of_surjective (f : M →ₗ[R] M') (h : Surjective f) :
    Module.rank R (LinearMap.range f) = Module.rank R M' := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    h : Function.Surjective ⇑f
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (LinearMap.range f) x)) ( …
  -/
  rw [LinearMap.range_eq_top.2 h, rank_top]
  /-
    🎉 no goals
  -/


theorem Submodule.rank_le (s : Submodule R M) : Module.rank R s ≤ Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Submodule R M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem s x)) (Module.rank R M)
  -/
  rw [← rank_top R M]
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Submodule R M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem s x)) (Module.rank R ( …
  -/
  exact rank_mono le_top
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-02")] alias rank_submodule_le := Submodule.rank_le


theorem LinearMap.lift_rank_le_of_surjective (f : M →ₗ[R] M') (h : Surjective f) :
    lift.{v} (Module.rank R M') ≤ lift.{v'} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    h : Function.Surjective ⇑f
    ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R M')) (Cardinal.lift.{v', v} (Mod …
  -/
  rw [← rank_range_of_surjective f h]
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    h : Function.Surjective ⇑f
    ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank R (Subtype fun x => Membership.mem …
  -/
  apply lift_rank_range_le
  /-
    🎉 no goals
  -/


theorem LinearMap.rank_le_of_surjective (f : M →ₗ[R] M₁) (h : Surjective f) :
    Module.rank R M₁ ≤ Module.rank R M := by
  /-
    R : Type u
    M M₁ : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    f : LinearMap (RingHom.id R) M M₁
    h : Function.Surjective ⇑f
    ⊢ LE.le (Module.rank R M₁) (Module.rank R M)
  -/
  rw [← rank_range_of_surjective f h]
  /-
    R : Type u
    M M₁ : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    f : LinearMap (RingHom.id R) M M₁
    h : Function.Surjective ⇑f
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (LinearMap.range f) x) …
  -/
  apply rank_range_le
  /-
    🎉 no goals
  -/


@[nontriviality, simp]
theorem rank_subsingleton [Subsingleton R] : Module.rank R M = 1 := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton R
    ⊢ Eq (Module.rank R M) 1
  -/
  haveI := Module.subsingleton R M
  have : Nonempty { s : Set M // LinearIndependent R ((↑) : s → M) } :=
    ⟨⟨∅, linearIndependent_empty _ _⟩⟩
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton R
    this✝ : Subsingleton M
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    ⊢ Eq (Module.rank R M) 1
  -/
  rw [Module.rank_def, ciSup_eq_of_forall_le_of_forall_lt_exists_gt]
    /-
      case h₁
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      ⊢ ∀ (i : Subtype fun s => LinearIndependent R Subtype.val), LE.le (Cardinal.mk …
    -/
  · rintro ⟨s, hs⟩
    /-
      case h₁.mk
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ LE.le (Cardinal.mk ↑↑⟨s, hs⟩) 1
    -/
    rw [Cardinal.mk_le_one_iff_set_subsingleton]
    /-
      case h₁.mk
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ (↑⟨s, hs⟩).Subsingleton
    -/
    apply subsingleton_of_subsingleton
    /-
      🎉 no goals
    -/
  /-
    case h₂
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton R
    this✝ : Subsingleton M
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    ⊢ ∀ (w : Cardinal.{v}), LT.lt w 1 → Exists fun i => LT.lt w (Cardinal.mk ↑↑i)
  -/
  intro w hw
  /-
    case h₂
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton R
    this✝ : Subsingleton M
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    w : Cardinal.{v}
    hw : LT.lt w 1
    ⊢ Exists fun i => LT.lt w (Cardinal.mk ↑↑i)
  -/
  refine ⟨⟨{0}, ?_⟩, ?_⟩
    /-
      case h₂.refine_1
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      w : Cardinal.{v}
      hw : LT.lt w 1
      ⊢ LinearIndependent R Subtype.val
    -/
  · rw [linearIndependent_iff']
    /-
      case h₂.refine_1
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      w : Cardinal.{v}
      hw : LT.lt w 1
      ⊢ ∀ (s : Finset (Subtype fun x => Membership.mem (Singleton.singleton 0) x)) ( …
    -/
    subsingleton
    /-
      🎉 no goals
    -/
    /-
      case h₂.refine_2
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Subsingleton R
      this✝ : Subsingleton M
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      w : Cardinal.{v}
      hw : LT.lt w 1
      ⊢ LT.lt w (Cardinal.mk ↑↑⟨Singleton.singleton 0, ⋯⟩)
    -/
  · exact hw.trans_eq (Cardinal.mk_singleton _).symm
    /-
      🎉 no goals
    -/


lemma rank_le_of_isSMulRegular {S : Type*} [CommSemiring S] [Algebra S R] [Module S M]
    [IsScalarTower S R M] (L L' : Submodule R M) {s : S} (hr : IsSMulRegular M s)
    (h : ∀ x ∈ L, s • x ∈ L') :
    Module.rank R L ≤ Module.rank R L' :=
  ((Algebra.lsmul S R M s).restrict h).rank_le_of_injective <|
                   /-
                     R : Type u
                     M : Type v
                     inst✝⁶ : Ring R
                     inst✝⁵ : AddCommGroup M
                     inst✝⁴ : Module R M
                     S : Type u_1
                     inst✝³ : CommSemiring S
                     inst✝² : Algebra S R
                     inst✝¹ : Module S M
                     inst✝ : IsScalarTower S R M
                     L L' : Submodule R M
                     s : S
                     hr : IsSMulRegular M s
                     h✝ : ∀ (x : M), Membership.mem L x → Membership.mem L' (HSMul.hSMul s x)
                     x✝¹ x✝ : Subtype fun x => Membership.mem L x
                     h : Eq ((LinearMap.restrict ((Algebra.lsmul S R M) s) h✝) x✝¹) ((LinearMap.res …
                     ⊢ Eq x✝¹ x✝
                   -/
    fun _ _ h ↦ by simpa using hr (Subtype.ext_iff.mp h)
                   /-
                     🎉 no goals
                   -/


@[deprecated (since := "2024-11-21")]
alias rank_le_of_smul_regular := rank_le_of_isSMulRegular


