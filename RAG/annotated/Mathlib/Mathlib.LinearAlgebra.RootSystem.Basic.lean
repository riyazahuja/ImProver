private theorem exist_eq_reflection_of_mapsTo  :
    ∃ k, root k = (preReflection (root i) (p.flip (coroot i))) (root j) :=
  h i (mem_range_self j)


private theorem choose_choose_eq_of_mapsTo :
    (exist_eq_reflection_of_mapsTo p root coroot i
      (exist_eq_reflection_of_mapsTo p root coroot i j h).choose h).choose = j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    i j : ι
    h : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (coro …
    hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
    ⊢ Eq ⋯.choose j
  -/
  refine root.injective ?_
  rw [(exist_eq_reflection_of_mapsTo p root coroot i _ h).choose_spec,
    (exist_eq_reflection_of_mapsTo p root coroot i j h).choose_spec]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    i j : ι
    h : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (coro …
    hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
    ⊢ Eq ((Module.preReflection (root i) (p.flip (coroot i))) ((Module.preReflecti …
  -/
  apply involutive_preReflection (x := root i) (hp i)
  /-
    🎉 no goals
  -/


/-- The bijection on the indexing set induced by reflection. -/
@[simps]
protected def equiv_of_mapsTo :
    ι ≃ ι where
  toFun j := (exist_eq_reflection_of_mapsTo p root coroot i j h).choose
  invFun j := (exist_eq_reflection_of_mapsTo p root coroot i j h).choose
  left_inv j := choose_choose_eq_of_mapsTo p root coroot i j h hp
  right_inv j := choose_choose_eq_of_mapsTo p root coroot i j h hp


lemma infinite_of_linearIndependent_coxeterWeight_four [CharZero R] [NoZeroSMulDivisors ℤ M]
    (P : RootPairing ι R M N) (i j : ι) (hl : LinearIndependent R ![P.root i, P.root j])
    (hc : P.coxeterWeight i j = 4) : Infinite ι := by
  refine (infinite_range_iff (Embedding.injective P.root)).mp (Infinite.mono ?_
    ((infinite_range_reflection_reflection_iterate_iff (P.coroot_root_two i)
    (P.coroot_root_two j) ?_).mpr ?_))
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
      hc : Eq (P.coxeterWeight i j) 4
      ⊢ HasSubset.Subset (Set.range fun n => Nat.iterate (⇑((Module.reflection ⋯).tr …
    -/
  · rw [range_subset_iff]
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
      hc : Eq (P.coxeterWeight i j) 4
      ⊢ ∀ (y : Nat), Membership.mem (Set.range ⇑P.root) (Nat.iterate (⇑((Module.refl …
    -/
    intro n
    rw [← IsFixedPt.image_iterate ((bijOn_reflection_of_mapsTo (P.coroot_root_two i)
      (P.mapsTo_reflection_root i)).comp (bijOn_reflection_of_mapsTo (P.coroot_root_two j)
      (P.mapsTo_reflection_root j))).image_eq n]
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
      hc : Eq (P.coxeterWeight i j) 4
      n : Nat
      ⊢ Membership.mem (Set.image (Nat.iterate (Function.comp ⇑(Module.reflection ⋯) …
    -/
    exact mem_image_of_mem _ (mem_range_self j)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
      hc : Eq (P.coxeterWeight i j) 4
      ⊢ Eq (HMul.hMul ((P.toLin.flip (P.coroot i)) (P.root j)) ((P.toLin.flip (P.cor …
    -/
  · rw [coroot_root_eq_pairing, coroot_root_eq_pairing, ← hc, mul_comm, coxeterWeight]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
      hc : Eq (P.coxeterWeight i j) 4
      ⊢ Ne (HSMul.hSMul ((P.toLin.flip (P.coroot i)) (P.root j)) (P.root i)) (HSMul. …
    -/
  · rw [LinearIndependent.pair_iff] at hl
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hl : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s (P.root i)) (HSMul.hSMul t (P.r …
      hc : Eq (P.coxeterWeight i j) 4
      ⊢ Ne (HSMul.hSMul ((P.toLin.flip (P.coroot i)) (P.root j)) (P.root i)) (HSMul. …
    -/
    specialize hl (P.pairing j i) (-2)
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hc : Eq (P.coxeterWeight i j) 4
      hl : Eq (HAdd.hAdd (HSMul.hSMul (P.pairing j i) (P.root i)) (HSMul.hSMul (-2)  …
      ⊢ Ne (HSMul.hSMul ((P.toLin.flip (P.coroot i)) (P.root j)) (P.root i)) (HSMul. …
    -/
    simp only [neg_smul, neg_eq_zero, OfNat.ofNat_ne_zero, and_false, imp_false] at hl
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hc : Eq (P.coxeterWeight i j) 4
      hl : Not (Eq (HAdd.hAdd (HSMul.hSMul (P.pairing j i) (P.root i)) (Neg.neg (HSM …
      ⊢ Ne (HSMul.hSMul ((P.toLin.flip (P.coroot i)) (P.root j)) (P.root i)) (HSMul. …
    -/
    rw [ne_eq, coroot_root_eq_pairing, ← sub_eq_zero, sub_eq_add_neg]
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors Int M
      P : RootPairing ι R M N
      i j : ι
      hc : Eq (P.coxeterWeight i j) 4
      hl : Not (Eq (HAdd.hAdd (HSMul.hSMul (P.pairing j i) (P.root i)) (Neg.neg (HSM …
      ⊢ Not (Eq (HAdd.hAdd (HSMul.hSMul (P.pairing j i) (P.root i)) (Neg.neg (HSMul. …
    -/
    exact hl
    /-
      🎉 no goals
    -/


lemma coxeterWeight_ne_four_of_linearIndependent [CharZero R] [NoZeroSMulDivisors ℤ M]
    (hl : LinearIndependent R ![P.root i, P.root j]) :
    P.coxeterWeight i j ≠ 4 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    P : RootPairing ι R M N
    i j : ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors Int M
    hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
    ⊢ Ne (P.coxeterWeight i j) 4
  -/
  intro contra
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    P : RootPairing ι R M N
    i j : ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors Int M
    hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
    contra : Eq (P.coxeterWeight i j) 4
    ⊢ False
  -/
  have := P.infinite_of_linearIndependent_coxeterWeight_four i j hl contra
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    P : RootPairing ι R M N
    i j : ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors Int M
    hl : LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P.root j) …
    contra : Eq (P.coxeterWeight i j) 4
    this : Infinite ι
    ⊢ False
  -/
  exact not_finite ι
  /-
    🎉 no goals
  -/


/-- Even though the roots may not span, coroots are distinguished by their pairing with the
roots. The proof depends crucially on the fact that there are finitely-many roots.

Modulo trivial generalisations, this statement is exactly Lemma 1.1.4 on page 87 of SGA 3 XXI. -/
lemma injOn_dualMap_subtype_span_root_coroot [NoZeroSMulDivisors ℤ M] :
    InjOn ((span R (range P.root)).subtype.dualMap ∘ₗ P.toLin.flip) (range P.coroot) := by
  have := injOn_dualMap_subtype_span_range_range (finite_range P.root)
    (c := P.toLin.flip ∘ P.coroot) P.root_coroot_two P.mapsTo_reflection_root
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Finite ι
    P : RootPairing ι R M N
    inst✝ : NoZeroSMulDivisors Int M
    this : Set.InjOn (⇑(Submodule.span R (Set.range ⇑P.root)).subtype.dualMap) (Se …
    ⊢ Set.InjOn (⇑((Submodule.span R (Set.range ⇑P.root)).subtype.dualMap.comp P.t …
  -/
  rintro - ⟨i, rfl⟩ - ⟨j, rfl⟩ hij
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Finite ι
    P : RootPairing ι R M N
    inst✝ : NoZeroSMulDivisors Int M
    this : Set.InjOn (⇑(Submodule.span R (Set.range ⇑P.root)).subtype.dualMap) (Se …
    i j : ι
    hij : Eq (((Submodule.span R (Set.range ⇑P.root)).subtype.dualMap.comp P.toLin …
    ⊢ Eq (P.coroot i) (P.coroot j)
  -/
  exact P.bijectiveRight.injective <| this (mem_range_self i) (mem_range_self j) hij
  /-
    🎉 no goals
  -/


/-- In characteristic zero if there is no torsion, the correspondence between roots and coroots is
unique.

Formally, the point is that the hypothesis `hc` depends only on the range of the coroot mappings. -/
@[ext]
protected lemma ext [CharZero R] [NoZeroSMulDivisors R M]
    {P₁ P₂ : RootPairing ι R M N}
    (he : P₁.toPerfectPairing = P₂.toPerfectPairing)
    (hr : P₁.root = P₂.root)
    (hc : range P₁.coroot = range P₂.coroot) :
    P₁ = P₂ := by
  have hp (hc' : P₁.coroot = P₂.coroot) : P₁.reflection_perm = P₂.reflection_perm := by
    ext i j
    refine P₁.root.injective ?_
    conv_rhs => rw [hr]
    simp only [root_reflection_perm, reflection_apply, coroot']
    simp only [hr, he, hc']
  suffices P₁.coroot = P₂.coroot by
    cases' P₁ with p₁; cases' P₂ with p₂; cases p₁; cases p₂; congr; exact hp this
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootPairing ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
    hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
    ⊢ Eq P₁.coroot P₂.coroot
  -/
  have := NoZeroSMulDivisors.int_of_charZero R M
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootPairing ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
    hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
    this : NoZeroSMulDivisors Int M
    ⊢ Eq P₁.coroot P₂.coroot
  -/
  ext i
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootPairing ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
    hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
    this : NoZeroSMulDivisors Int M
    i : ι
    ⊢ Eq (P₁.coroot i) (P₂.coroot i)
  -/
  apply P₁.injOn_dualMap_subtype_span_root_coroot (mem_range_self i) (hc ▸ mem_range_self i)
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootPairing ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
    hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
    this : NoZeroSMulDivisors Int M
    i : ι
    ⊢ Eq (((Submodule.span R (Set.range ⇑P₁.root)).subtype.dualMap.comp P₁.toLin.f …
  -/
  simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, comp_apply]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootPairing ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
    hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
    this : NoZeroSMulDivisors Int M
    i : ι
    ⊢ Eq ((Submodule.span R (Set.range ⇑P₁.root)).subtype.dualMap (P₁.toLin.flip ( …
  -/
  apply Dual.eq_of_preReflection_mapsTo' (P₁.ne_zero i) (finite_range P₁.root)
    /-
      case h.hx'
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootPairing ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
      hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
      this : NoZeroSMulDivisors Int M
      i : ι
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑P₁.root)) (P₁.root i)
    -/
  · exact Submodule.subset_span (mem_range_self i)
    /-
      🎉 no goals
    -/
    /-
      case h.hf₁
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootPairing ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
      hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
      this : NoZeroSMulDivisors Int M
      i : ι
      ⊢ Eq ((P₁.toLin.flip (P₁.coroot i)) (P₁.root i)) 2
    -/
  · exact P₁.coroot_root_two i
    /-
      🎉 no goals
    -/
    /-
      case h.hf₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootPairing ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
      hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
      this : NoZeroSMulDivisors Int M
      i : ι
      ⊢ Set.MapsTo (⇑(Module.preReflection (P₁.root i) (P₁.toLin.flip (P₁.coroot i)) …
    -/
  · exact P₁.mapsTo_reflection_root i
    /-
      🎉 no goals
    -/
    /-
      case h.hg₁
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootPairing ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
      hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
      this : NoZeroSMulDivisors Int M
      i : ι
      ⊢ Eq ((P₁.toLin.flip (P₂.coroot i)) (P₁.root i)) 2
    -/
  · exact hr ▸ he ▸ P₂.coroot_root_two i
    /-
      🎉 no goals
    -/
    /-
      case h.hg₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootPairing ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      hc : Eq (Set.range ⇑P₁.coroot) (Set.range ⇑P₂.coroot)
      hp : Eq P₁.coroot P₂.coroot → Eq P₁.reflection_perm P₂.reflection_perm
      this : NoZeroSMulDivisors Int M
      i : ι
      ⊢ Set.MapsTo (⇑(Module.preReflection (P₁.root i) (P₁.toLin.flip (P₂.coroot i)) …
    -/
  · exact hr ▸ he ▸ P₂.mapsTo_reflection_root i
    /-
      🎉 no goals
    -/


private lemma coroot_eq_coreflection_of_root_eq' [CharZero R] [NoZeroSMulDivisors R M]
    (p : PerfectPairing R M N)
    (root : ι ↪ M)
    (coroot : ι ↪ N)
    (hp : ∀ i, p (root i) (coroot i) = 2)
    (hr : ∀ i, MapsTo (preReflection (root i) (p.flip (coroot i))) (range root) (range root))
    (hc : ∀ i, MapsTo (preReflection (coroot i) (p (root i))) (range coroot) (range coroot))
    {i j k : ι} (hk : root k = preReflection (root i) (p.flip (coroot i)) (root j)) :
    coroot k = preReflection (coroot i) (p (root i)) (coroot j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    hk : Eq (root k) ((Module.preReflection (root i) (p.flip (coroot i))) (root j))
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
  -/
  set α := root i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    hk : Eq (root k) ((Module.preReflection α (p.flip (coroot i))) (root j))
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p α)) (coroot j))
  -/
  set β := root j
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    hk : Eq (root k) ((Module.preReflection α (p.flip (coroot i))) β)
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p α)) (coroot j))
  -/
  set α' := coroot i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    hk : Eq (root k) ((Module.preReflection α (p.flip α')) β)
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) (coroot j))
  -/
  set β' := coroot j
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    hk : Eq (root k) ((Module.preReflection α (p.flip α')) β)
    β' : N := coroot j
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  set sα := preReflection α (p.flip α')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  set sβ := preReflection β (p.flip β')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  let sα' := preReflection α' (p α)
  have hij : preReflection (sα β) (p.flip (sα' β')) = sα ∘ₗ sβ ∘ₗ sα := by
    ext
    have hpi : (p.flip (coroot i)) (root i) = 2 := by rw [PerfectPairing.flip_apply_apply, hp i]
    simp [α, β, α', β', sα, sβ, sα', ← preReflection_preReflection β (p.flip β') hpi,
      preReflection_apply]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  have hk₀ : root k ≠ 0 := fun h ↦ by simpa [h, ← PerfectPairing.toLin_apply] using hp k
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  obtain ⟨l, hl⟩ := hc i (mem_range_self j)
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  rw [← hl]
  have hkl : (p.flip (coroot l)) (root k) = 2 := by
    simp only [hl, preReflection_apply, hk, PerfectPairing.flip_apply_apply, map_sub, hp j,
      map_smul, smul_eq_mul, hp i, mul_sub, sα, α, α', β, mul_two, mul_add, LinearMap.sub_apply,
      LinearMap.smul_apply]
    rw [mul_comm (p (root i) (coroot j))]
    abel
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    hkl : Eq ((p.flip (coroot l)) (root k)) 2
    ⊢ Eq (coroot k) (coroot l)
  -/
  suffices p.flip (coroot k) = p.flip (coroot l) from p.bijectiveRight.injective this
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    hkl : Eq ((p.flip (coroot l)) (root k)) 2
    ⊢ Eq (p.flip (coroot k)) (p.flip (coroot l))
  -/
  have _i : NoZeroSMulDivisors ℤ M := NoZeroSMulDivisors.int_of_charZero R M
  have := injOn_dualMap_subtype_span_range_range (finite_range root)
    (c := p.flip ∘ coroot) hp hr
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    hkl : Eq ((p.flip (coroot l)) (root k)) 2
    _i : NoZeroSMulDivisors Int M
    this : Set.InjOn (⇑(Submodule.span R (Set.range ⇑root)).subtype.dualMap) (Set. …
    ⊢ Eq (p.flip (coroot k)) (p.flip (coroot l))
  -/
  apply this (mem_range_self k) (mem_range_self l)
  refine Dual.eq_of_preReflection_mapsTo' hk₀ (finite_range root)
    (Submodule.subset_span <| mem_range_self k) (hp k) (hr k) hkl ?_
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    hkl : Eq ((p.flip (coroot l)) (root k)) 2
    _i : NoZeroSMulDivisors Int M
    this : Set.InjOn (⇑(Submodule.span R (Set.range ⇑root)).subtype.dualMap) (Set. …
    ⊢ Set.MapsTo (⇑(Module.preReflection (root k) (Function.comp (⇑p.flip) (⇑coroo …
  -/
  rw [comp_apply, hl, hk, hij]
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.flip (sα' β'))) (LinearMap.comp sα (L …
    hk₀ : Ne (root k) 0
    l : ι
    hl : Eq (coroot l) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
    hkl : Eq ((p.flip (coroot l)) (root k)) 2
    _i : NoZeroSMulDivisors Int M
    this : Set.InjOn (⇑(Submodule.span R (Set.range ⇑root)).subtype.dualMap) (Set. …
    ⊢ Set.MapsTo (⇑(LinearMap.comp sα (LinearMap.comp sβ sα))) (Set.range ⇑root) ( …
  -/
  exact (hr i).comp <| (hr j).comp (hr i)
  /-
    🎉 no goals
  -/


/-- In characteristic zero if there is no torsion, to check that two finite families of roots and
coroots form a root pairing, it is sufficient to check that they are stable under reflections. -/
def mk' [Finite ι] [CharZero R] [NoZeroSMulDivisors R M]
    (p : PerfectPairing R M N)
    (root : ι ↪ M)
    (coroot : ι ↪ N)
    (hp : ∀ i, p (root i) (coroot i) = 2)
    (hr : ∀ i, MapsTo (preReflection (root i) (p.flip (coroot i))) (range root) (range root))
    (hc : ∀ i, MapsTo (preReflection (coroot i) (p (root i))) (range coroot) (range coroot)) :
    RootPairing ι R M N where
  toPerfectPairing := p
  root := root
  coroot := coroot
  root_coroot_two := hp
  reflection_perm i := RootPairing.equiv_of_mapsTo p root coroot i hr hp
  reflection_perm_root i j := by
    rw [equiv_of_mapsTo_apply, (exist_eq_reflection_of_mapsTo p root coroot i j hr).choose_spec,
      preReflection_apply, PerfectPairing.flip_apply_apply]
  reflection_perm_coroot i j := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      inst✝³ : Finite ι
      P : RootPairing ι R M N
      i✝ j✝ : ι
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
      hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
      hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
      i j : ι
      ⊢ Eq (HSub.hSub (coroot j) (HSMul.hSMul ((p (root i)) (coroot j)) (coroot i))) …
    -/
    refine (coroot_eq_coreflection_of_root_eq' p root coroot hp hr hc ?_).symm
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      inst✝³ : Finite ι
      P : RootPairing ι R M N
      i✝ j✝ : ι
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
      hr : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
      hc : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) ( …
      i j : ι
      ⊢ Eq (root (((fun i => RootPairing.equiv_of_mapsTo p root coroot i hr hp) i) j …
    -/
    rw [equiv_of_mapsTo_apply, (exist_eq_reflection_of_mapsTo p root coroot i j hr).choose_spec]
    /-
      🎉 no goals
    -/


/-- In characteristic zero if there is no torsion, a finite root system is determined entirely by
its roots. -/
@[ext]
protected lemma ext [CharZero R] [NoZeroSMulDivisors R M]
    {P₁ P₂ : RootSystem ι R M N}
    (he : P₁.toPerfectPairing = P₂.toPerfectPairing)
    (hr : P₁.root = P₂.root) :
    P₁ = P₂ := by
  suffices ∀ P₁ P₂ : RootSystem ι R M N, P₁.toPerfectPairing = P₂.toPerfectPairing →
      P₁.root = P₂.root → range P₁.coroot ⊆ range P₂.coroot by
    have h₁ := this P₁ P₂ he hr
    have h₂ := this P₂ P₁ he.symm hr.symm
    cases' P₁ with P₁
    cases' P₂ with P₂
    congr
    exact RootPairing.ext he hr (le_antisymm h₁ h₂)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootSystem ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    ⊢ ∀ (P₁ P₂ : RootSystem ι R M N), Eq P₁.toPerfectPairing P₂.toPerfectPairing → …
  -/
  clear! P₁ P₂
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    ⊢ ∀ (P₁ P₂ : RootSystem ι R M N), Eq P₁.toPerfectPairing P₂.toPerfectPairing → …
  -/
  rintro P₁ P₂ he hr - ⟨i, rfl⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootSystem ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    i : ι
    ⊢ Membership.mem (Set.range ⇑P₂.coroot) (P₁.coroot i)
  -/
  use i
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootSystem ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    i : ι
    ⊢ Eq (P₂.coroot i) (P₁.coroot i)
  -/
  apply P₁.bijectiveRight.injective
  /-
    case h.a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    P₁ P₂ : RootSystem ι R M N
    he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
    hr : Eq P₁.root P₂.root
    i : ι
    ⊢ Eq (P₁.toLin.flip (P₂.coroot i)) (P₁.toLin.flip (P₁.coroot i))
  -/
  apply Dual.eq_of_preReflection_mapsTo (P₁.ne_zero i) (finite_range P₁.root) P₁.span_eq_top
    /-
      case h.a.hf₁
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootSystem ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      i : ι
      ⊢ Eq ((P₁.toLin.flip (P₂.coroot i)) (P₁.root i)) 2
    -/
  · exact hr ▸ he ▸ P₂.coroot_root_two i
    /-
      🎉 no goals
    -/
    /-
      case h.a.hf₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootSystem ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      i : ι
      ⊢ Set.MapsTo (⇑(Module.preReflection (P₁.root i) (P₁.toLin.flip (P₂.coroot i)) …
    -/
  · exact hr ▸ he ▸ P₂.mapsTo_reflection_root i
    /-
      🎉 no goals
    -/
    /-
      case h.a.hg₁
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootSystem ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      i : ι
      ⊢ Eq ((P₁.toLin.flip (P₁.coroot i)) (P₁.root i)) 2
    -/
  · exact P₁.coroot_root_two i
    /-
      🎉 no goals
    -/
    /-
      case h.a.hg₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      P₁ P₂ : RootSystem ι R M N
      he : Eq P₁.toPerfectPairing P₂.toPerfectPairing
      hr : Eq P₁.root P₂.root
      i : ι
      ⊢ Set.MapsTo (⇑(Module.preReflection (P₁.root i) (P₁.toLin.flip (P₁.coroot i)) …
    -/
  · exact P₁.mapsTo_reflection_root i
    /-
      🎉 no goals
    -/


private lemma coroot_eq_coreflection_of_root_eq_of_span_eq_top [CharZero R] [NoZeroSMulDivisors R M]
    (p : PerfectPairing R M N)
    (root : ι ↪ M)
    (coroot : ι ↪ N)
    (hp : ∀ i, p (root i) (coroot i) = 2)
    (hs : ∀ i, MapsTo (preReflection (root i) (p.flip (coroot i))) (range root) (range root))
    (hsp : span R (range root) = ⊤)
    {i j k : ι} (hk : root k = preReflection (root i) (p.flip (coroot i)) (root j)) :
    coroot k = preReflection (coroot i) (p (root i)) (coroot j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    hk : Eq (root k) ((Module.preReflection (root i) (p.flip (coroot i))) (root j))
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p (root i))) (coroot j))
  -/
  set α := root i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    hk : Eq (root k) ((Module.preReflection α (p.flip (coroot i))) (root j))
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p α)) (coroot j))
  -/
  set β := root j
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    hk : Eq (root k) ((Module.preReflection α (p.flip (coroot i))) β)
    ⊢ Eq (coroot k) ((Module.preReflection (coroot i) (p α)) (coroot j))
  -/
  set α' := coroot i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    hk : Eq (root k) ((Module.preReflection α (p.flip α')) β)
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) (coroot j))
  -/
  set β' := coroot j
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    hk : Eq (root k) ((Module.preReflection α (p.flip α')) β)
    β' : N := coroot j
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  set sα := preReflection α (p.flip α')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  set sβ := preReflection β (p.flip β')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  let sα' := preReflection α' (p α)
  have hij : preReflection (sα β) (p.toLin.flip (sα' β')) = sα ∘ₗ sβ ∘ₗ sα := by
    ext
    have hpi : (p.flip (coroot i)) (root i) = 2 := by rw [PerfectPairing.flip_apply_apply, hp i]
    simp [α, β, α', β', sα, sβ, sα', ← preReflection_preReflection β (p.flip β') hpi,
      preReflection_apply] -- v4.7.0-rc1 issues
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  have hk₀ : root k ≠ 0 := fun h ↦ by simpa [h, ← PerfectPairing.toLin_apply] using hp k
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
    hk₀ : Ne (root k) 0
    ⊢ Eq (coroot k) ((Module.preReflection α' (p α)) β')
  -/
  apply p.bijectiveRight.injective
  /-
    case a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : CharZero R
    inst✝ : NoZeroSMulDivisors R M
    p : PerfectPairing R M N
    root : Function.Embedding ι M
    coroot : Function.Embedding ι N
    hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
    hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
    hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
    i j k : ι
    α : M := root i
    β : M := root j
    α' : N := coroot i
    β' : N := coroot j
    sα : Module.End R M := Module.preReflection α (p.flip α')
    hk : Eq (root k) (sα β)
    sβ : Module.End R M := Module.preReflection β (p.flip β')
    sα' : Module.End R N := Module.preReflection α' (p α)
    hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
    hk₀ : Ne (root k) 0
    ⊢ Eq (p.toLin.flip (coroot k)) (p.toLin.flip ((Module.preReflection α' (p α))  …
  -/
  apply Dual.eq_of_preReflection_mapsTo hk₀ (finite_range root) hsp (hp k) (hs k)
  · simp [map_sub, α, β, α', β', sα, sβ, sα', hk, preReflection_apply, hp i, hp j, mul_two,
      mul_comm (p α β')]
    /-
      case a.hg₁
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j k : ι
      α : M := root i
      β : M := root j
      α' : N := coroot i
      β' : N := coroot j
      sα : Module.End R M := Module.preReflection α (p.flip α')
      hk : Eq (root k) (sα β)
      sβ : Module.End R M := Module.preReflection β (p.flip β')
      sα' : Module.End R N := Module.preReflection α' (p α)
      hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
      hk₀ : Ne (root k) 0
      ⊢ Eq (HSub.hSub (HSub.hSub 2 (HMul.hMul ((p (root j)) (coroot i)) ((p (root i) …
    -/
    ring -- v4.7.0-rc1 issues
    /-
      🎉 no goals
    -/
    /-
      case a.hg₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j k : ι
      α : M := root i
      β : M := root j
      α' : N := coroot i
      β' : N := coroot j
      sα : Module.End R M := Module.preReflection α (p.flip α')
      hk : Eq (root k) (sα β)
      sβ : Module.End R M := Module.preReflection β (p.flip β')
      sα' : Module.End R N := Module.preReflection α' (p α)
      hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
      hk₀ : Ne (root k) 0
      ⊢ Set.MapsTo (⇑(Module.preReflection (root k) (p.toLin.flip ((Module.preReflec …
    -/
  · rw [hk, hij]
    /-
      case a.hg₂
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.flip (coroot i) …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j k : ι
      α : M := root i
      β : M := root j
      α' : N := coroot i
      β' : N := coroot j
      sα : Module.End R M := Module.preReflection α (p.flip α')
      hk : Eq (root k) (sα β)
      sβ : Module.End R M := Module.preReflection β (p.flip β')
      sα' : Module.End R N := Module.preReflection α' (p α)
      hij : Eq (Module.preReflection (sα β) (p.toLin.flip (sα' β'))) (LinearMap.comp …
      hk₀ : Ne (root k) 0
      ⊢ Set.MapsTo (⇑(LinearMap.comp sα (LinearMap.comp sβ sα))) (Set.range ⇑root) ( …
    -/
    exact (hs i).comp <| (hs j).comp (hs i)
    /-
      🎉 no goals
    -/


/-- In characteristic zero if there is no torsion, to check that a finite family of roots form a
root system, we do not need to check that the coroots are stable under reflections since this
follows from the corresponding property for the roots. -/
def mk' [CharZero R] [NoZeroSMulDivisors R M]
    (p : PerfectPairing R M N)
    (root : ι ↪ M)
    (coroot : ι ↪ N)
    (hp : ∀ i, p.toLin (root i) (coroot i) = 2)
    (hs : ∀ i, MapsTo (preReflection (root i) (p.toLin.flip (coroot i))) (range root) (range root))
    (hsp : span R (range root) = ⊤) :
    RootSystem ι R M N where
  span_eq_top := hsp
  toRootPairing := RootPairing.mk' p root coroot hp hs <| by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      P : RootSystem ι R M N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (cor …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      ⊢ ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (coroot i) (p (root i)))) (Set …
    -/
    rintro i - ⟨j, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      P : RootSystem ι R M N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (cor …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j : ι
      ⊢ Membership.mem (Set.range ⇑coroot) ((Module.preReflection (coroot i) (p (roo …
    -/
    use RootPairing.equiv_of_mapsTo p root coroot i hs hp j
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      P : RootSystem ι R M N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (cor …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j : ι
      ⊢ Eq (coroot ((RootPairing.equiv_of_mapsTo p root coroot i hs hp) j)) ((Module …
    -/
    refine (coroot_eq_coreflection_of_root_eq_of_span_eq_top p root coroot hp hs hsp ?_)
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁷ : CommRing R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R N
      inst✝² : Finite ι
      P : RootSystem ι R M N
      inst✝¹ : CharZero R
      inst✝ : NoZeroSMulDivisors R M
      p : PerfectPairing R M N
      root : Function.Embedding ι M
      coroot : Function.Embedding ι N
      hp : ∀ (i : ι), Eq ((p.toLin (root i)) (coroot i)) 2
      hs : ∀ (i : ι), Set.MapsTo (⇑(Module.preReflection (root i) (p.toLin.flip (cor …
      hsp : Eq (Submodule.span R (Set.range ⇑root)) Top.top
      i j : ι
      ⊢ Eq (root ((RootPairing.equiv_of_mapsTo p root coroot i hs hp) j)) ((Module.p …
    -/
    rw [equiv_of_mapsTo_apply, (exist_eq_reflection_of_mapsTo  p root coroot i j hs).choose_spec]
    /-
      🎉 no goals
    -/


