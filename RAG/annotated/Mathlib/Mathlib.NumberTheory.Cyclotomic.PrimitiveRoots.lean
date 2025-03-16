/-- If `B` is an `n`-th cyclotomic extension of `A`, then `zeta n A B` is a primitive root of
unity in `B`. -/
noncomputable def zeta : B :=
  (exists_prim_root A <| Set.mem_singleton n : ∃ r : B, IsPrimitiveRoot r n).choose


/-- `zeta n A B` is a primitive `n`-th root of unity. -/
@[simp]
theorem zeta_spec : IsPrimitiveRoot (zeta n A B) n :=
  Classical.choose_spec (exists_prim_root A (Set.mem_singleton n) : ∃ r : B, IsPrimitiveRoot r n)


theorem aeval_zeta [IsDomain B] [NeZero ((n : ℕ) : B)] :
    aeval (zeta n A B) (cyclotomic n A) = 0 := by
  /-
    n : PNat
    A : Type w
    B : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝¹ : IsDomain B
    inst✝ : NeZero ↑↑n
    ⊢ Eq ((Polynomial.aeval (IsCyclotomicExtension.zeta n A B)) (Polynomial.cyclot …
  -/
  rw [aeval_def, ← eval_map, ← IsRoot.def, map_cyclotomic, isRoot_cyclotomic_iff]
  /-
    n : PNat
    A : Type w
    B : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝¹ : IsDomain B
    inst✝ : NeZero ↑↑n
    ⊢ IsPrimitiveRoot (IsCyclotomicExtension.zeta n A B) ↑n
  -/
  exact zeta_spec n A B
  /-
    🎉 no goals
  -/


theorem zeta_isRoot [IsDomain B] [NeZero ((n : ℕ) : B)] : IsRoot (cyclotomic n B) (zeta n A B) := by
  /-
    n : PNat
    A : Type w
    B : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝¹ : IsDomain B
    inst✝ : NeZero ↑↑n
    ⊢ (Polynomial.cyclotomic (↑n) B).IsRoot (IsCyclotomicExtension.zeta n A B)
  -/
  convert aeval_zeta n A B using 0
  /-
    case a
    n : PNat
    A : Type w
    B : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝¹ : IsDomain B
    inst✝ : NeZero ↑↑n
    ⊢ Iff ((Polynomial.cyclotomic (↑n) B).IsRoot (IsCyclotomicExtension.zeta n A B …
  -/
  rw [IsRoot.def, aeval_def, eval₂_eq_eval_map, map_cyclotomic]
  /-
    🎉 no goals
  -/


theorem zeta_pow : zeta n A B ^ (n : ℕ) = 1 :=
  (zeta_spec n A B).pow_eq_one


/-- The `PowerBasis` given by a primitive root `η`. -/
@[simps!]
protected noncomputable def powerBasis : PowerBasis K L :=
  -- this is purely an optimization
  letI pb := Algebra.adjoin.powerBasis <| (integral {n} K L).isIntegral ζ
  pb.map <| (Subalgebra.equivOfEq _ _ (IsCyclotomicExtension.adjoin_primitive_root_eq_top hζ)).trans
    Subalgebra.topEquiv


theorem powerBasis_gen_mem_adjoin_zeta_sub_one :
    (hζ.powerBasis K).gen ∈ adjoin K ({ζ - 1} : Set L) := by
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    ζ : L
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Membership.mem (Algebra.adjoin K (Singleton.singleton (HSub.hSub ζ 1))) (IsP …
  -/
  rw [powerBasis_gen, adjoin_singleton_eq_range_aeval, AlgHom.mem_range]
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    ζ : L
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Exists fun x => Eq ((Polynomial.aeval (HSub.hSub ζ 1)) x) ζ
  -/
  exact ⟨X + 1, by simp⟩
  /-
    🎉 no goals
  -/


/-- The `PowerBasis` given by `η - 1`. -/
@[simps!]
noncomputable def subOnePowerBasis : PowerBasis K L :=
  (hζ.powerBasis K).ofGenMemAdjoin
    (((integral {n} K L).isIntegral ζ).sub isIntegral_one)
    (hζ.powerBasis_gen_mem_adjoin_zeta_sub_one _)


/-- The equivalence between `L →ₐ[K] C` and `primitiveRoots n C` given by a primitive root `ζ`. -/
noncomputable def embeddingsEquivPrimitiveRoots (C : Type*) [CommRing C] [IsDomain C] [Algebra K C]
    (hirr : Irreducible (cyclotomic n K)) : (L →ₐ[K] C) ≃ primitiveRoots n C :=
  (hζ.powerBasis K).liftEquiv.trans
    { toFun := fun x => by
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powe …
          ⊢ Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
        -/
        haveI := IsCyclotomicExtension.neZero' n K L
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powe …
          this : NeZero ↑↑n
          ⊢ Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
        -/
        haveI hn := NeZero.of_noZeroSMulDivisors K C n
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powe …
          this : NeZero ↑↑n
          hn : NeZero ↑↑n
          ⊢ Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
        -/
        refine ⟨x.1, ?_⟩
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powe …
          this : NeZero ↑↑n
          hn : NeZero ↑↑n
          ⊢ Membership.mem (primitiveRoots (↑n) C) ↑x
        -/
        cases x
        rwa [mem_primitiveRoots n.pos, ← isRoot_cyclotomic_iff, IsRoot.def,
          ← map_cyclotomic _ (algebraMap K C), hζ.minpoly_eq_cyclotomic_of_irreducible hirr,
          ← eval₂_eq_eval_map, ← aeval_def]
      invFun := fun x => by
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
          ⊢ Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powerB …
        -/
        haveI := IsCyclotomicExtension.neZero' n K L
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
          this : NeZero ↑↑n
          ⊢ Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powerB …
        -/
        haveI hn := NeZero.of_noZeroSMulDivisors K C n
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
          this : NeZero ↑↑n
          hn : NeZero ↑↑n
          ⊢ Subtype fun y => Eq ((Polynomial.aeval y) (minpoly K (IsPrimitiveRoot.powerB …
        -/
        refine ⟨x.1, ?_⟩
        /-
          p n : PNat
          A : Type w
          B : Type z
          K : Type u
          L : Type v
          C✝ : Type w
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra A B
          inst✝⁸ : IsCyclotomicExtension (Singleton.singleton n) A B
          inst✝⁷ : Field K
          inst✝⁶ : CommRing L
          inst✝⁵ : IsDomain L
          inst✝⁴ : Algebra K L
          inst✝³ : IsCyclotomicExtension (Singleton.singleton n) K L
          ζ : L
          hζ : IsPrimitiveRoot ζ ↑n
          C : Type u_1
          inst✝² : CommRing C
          inst✝¹ : IsDomain C
          inst✝ : Algebra K C
          hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
          x : Subtype fun x => Membership.mem (primitiveRoots (↑n) C) x
          this : NeZero ↑↑n
          hn : NeZero ↑↑n
          ⊢ Eq ((Polynomial.aeval ↑x) (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen)) 0
        -/
        cases x
        rwa [aeval_def, eval₂_eq_eval_map, hζ.powerBasis_gen K, ←
          hζ.minpoly_eq_cyclotomic_of_irreducible hirr, map_cyclotomic, ← IsRoot.def,
          isRoot_cyclotomic_iff, ← mem_primitiveRoots n.pos]
      left_inv := fun _ => Subtype.ext rfl
      right_inv := fun _ => Subtype.ext rfl }

-- Porting note: renamed argument `φ`: "expected '_' or identifier"

@[simp]
theorem embeddingsEquivPrimitiveRoots_apply_coe (C : Type*) [CommRing C] [IsDomain C] [Algebra K C]
    (hirr : Irreducible (cyclotomic n K)) (φ' : L →ₐ[K] C) :
    (hζ.embeddingsEquivPrimitiveRoots C hirr φ' : C) = φ' ζ :=
  rfl


/-- If `Irreducible (cyclotomic n K)` (in particular for `K = ℚ`), then the `finrank` of a
cyclotomic extension is `n.totient`. -/
theorem finrank (hirr : Irreducible (cyclotomic n K)) : finrank K L = (n : ℕ).totient := by
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq (Module.finrank K L) (↑n).totient
  -/
  haveI := IsCyclotomicExtension.neZero' n K L
  rw [((zeta_spec n K L).powerBasis K).finrank, IsPrimitiveRoot.powerBasis_dim, ←
    (zeta_spec n K L).minpoly_eq_cyclotomic_of_irreducible hirr, natDegree_cyclotomic]


variable {L} in
/-- If `L` contains both a primitive `p`-th root of unity and `q`-th root of unity, and
`Irreducible (cyclotomic (lcm p q) K)` (in particular for `K = ℚ`), then the `finrank K L` is at
least `(lcm p q).totient`. -/
theorem _root_.IsPrimitiveRoot.lcm_totient_le_finrank [FiniteDimensional K L] {p q : ℕ} {x y : L}
    (hx : IsPrimitiveRoot x p) (hy : IsPrimitiveRoot y q)
    (hirr : Irreducible (cyclotomic (Nat.lcm p q) K)) :
    (Nat.lcm p q).totient ≤ Module.finrank K L := by
  /-
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  rcases Nat.eq_zero_or_pos p with (rfl | hppos)
    /-
      case inl
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : CommRing L
      inst✝² : IsDomain L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      q : Nat
      x y : L
      hy : IsPrimitiveRoot y q
      hx : IsPrimitiveRoot x 0
      hirr : Irreducible (Polynomial.cyclotomic (Nat.lcm 0 q) K)
      ⊢ LE.le (Nat.lcm 0 q).totient (Module.finrank K L)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  rcases Nat.eq_zero_or_pos q with (rfl | hqpos)
    /-
      case inr.inl
      K : Type u
      L : Type v
      inst✝⁴ : Field K
      inst✝³ : CommRing L
      inst✝² : IsDomain L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      p : Nat
      x y : L
      hx : IsPrimitiveRoot x p
      hppos : GT.gt p 0
      hy : IsPrimitiveRoot y 0
      hirr : Irreducible (Polynomial.cyclotomic (p.lcm 0) K)
      ⊢ LE.le (p.lcm 0).totient (Module.finrank K L)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  let z := x ^ (p / factorizationLCMLeft p q) * y ^ (q / factorizationLCMRight p q)
  /-
    case inr.inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  let k := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
  /-
    case inr.inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    k : PNat := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  have : IsPrimitiveRoot z k := hx.pow_mul_pow_lcm hy hppos.ne' hqpos.ne'
  /-
    case inr.inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    k : PNat := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
    this : IsPrimitiveRoot z ↑k
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  haveI := IsPrimitiveRoot.adjoin_isCyclotomicExtension K this
  /-
    case inr.inr
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    k : PNat := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
    this✝ : IsPrimitiveRoot z ↑k
    this : IsCyclotomicExtension (Singleton.singleton k) K (Subtype fun x => Membe …
    ⊢ LE.le (p.lcm q).totient (Module.finrank K L)
  -/
  convert Submodule.finrank_le (Subalgebra.toSubmodule (adjoin K {z}))
  /-
    case h.e'_3
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hirr : Irreducible (Polynomial.cyclotomic (p.lcm q) K)
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    k : PNat := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
    this✝ : IsPrimitiveRoot z ↑k
    this : IsCyclotomicExtension (Singleton.singleton k) K (Subtype fun x => Membe …
    ⊢ Eq (p.lcm q).totient (Module.finrank K (Subtype fun x => Membership.mem (Sub …
  -/
  rw [show Nat.lcm p q = (k : ℕ) from rfl] at hirr
  /-
    case h.e'_3
    K : Type u
    L : Type v
    inst✝⁴ : Field K
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    p q : Nat
    x y : L
    hx : IsPrimitiveRoot x p
    hy : IsPrimitiveRoot y q
    hppos : GT.gt p 0
    hqpos : GT.gt q 0
    z : L := HMul.hMul (HPow.hPow x (HDiv.hDiv p (p.factorizationLCMLeft q))) (HPo …
    k : PNat := PNat.lcm ⟨p, hppos⟩ ⟨q, hqpos⟩
    hirr : Irreducible (Polynomial.cyclotomic (↑k) K)
    this✝ : IsPrimitiveRoot z ↑k
    this : IsCyclotomicExtension (Singleton.singleton k) K (Subtype fun x => Membe …
    ⊢ Eq (p.lcm q).totient (Module.finrank K (Subtype fun x => Membership.mem (Sub …
  -/
  simpa using (IsCyclotomicExtension.finrank (Algebra.adjoin K {z}) hirr).symm
  /-
    🎉 no goals
  -/


variable (n) in
/-- If a `n`-th cyclotomic extension of `ℚ` contains a primitive `l`-th root of unity, then
`l ∣ 2 * n`. -/
theorem dvd_of_isCyclotomicExtension [IsCyclotomicExtension {n} ℚ K] {ζ : K}
    {l : ℕ} (hζ : IsPrimitiveRoot ζ l) (hl : l ≠ 0) : l ∣ 2 * n := by
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl : Ne l 0
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  have hl : NeZero l := ⟨hl⟩
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  have hroot := IsCyclotomicExtension.zeta_spec n ℚ K
  have key := IsPrimitiveRoot.lcm_totient_le_finrank hζ hroot
    (cyclotomic.irreducible_rat <| Nat.lcm_pos (Nat.pos_of_ne_zero hl.1) n.2)
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    key : LE.le (l.lcm ↑n).totient (Module.finrank Rat K)
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  rw [IsCyclotomicExtension.finrank K (cyclotomic.irreducible_rat n.2)] at key
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    key : LE.le (l.lcm ↑n).totient (↑n).totient
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  rcases _root_.dvd_lcm_right l n with ⟨r, hr⟩
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    key : LE.le (l.lcm ↑n).totient (↑n).totient
    r : Nat
    hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  have ineq := Nat.totient_super_multiplicative n r
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    key : LE.le (l.lcm ↑n).totient (↑n).totient
    r : Nat
    hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
    ineq : LE.le (HMul.hMul (↑n).totient r.totient) (HMul.hMul (↑n) r).totient
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  rw [← hr] at ineq
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    key : LE.le (l.lcm ↑n).totient (↑n).totient
    r : Nat
    hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
    ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  replace key := (mul_le_iff_le_one_right (Nat.totient_pos.2 n.2)).mp (le_trans ineq key)
  have rpos : 0 < r := by
    refine Nat.pos_of_ne_zero (fun h ↦ ?_)
    simp only [h, mul_zero, _root_.lcm_eq_zero_iff, PNat.ne_zero, or_false] at hr
    exact hl.1 hr
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    r : Nat
    hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
    ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
    key : LE.le r.totient 1
    rpos : LT.lt 0 r
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  replace key := (Nat.dvd_prime Nat.prime_two).1 (Nat.dvd_two_of_totient_le_one rpos key)
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ζ : K
    l : Nat
    hζ : IsPrimitiveRoot ζ l
    hl✝ : Ne l 0
    hl : NeZero l
    hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
    r : Nat
    hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
    ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
    rpos : LT.lt 0 r
    key : Or (Eq r 1) (Eq r 2)
    ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
  -/
  rcases key with (key | key)
    /-
      case intro.inl
      n : PNat
      K : Type u
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
      ζ : K
      l : Nat
      hζ : IsPrimitiveRoot ζ l
      hl✝ : Ne l 0
      hl : NeZero l
      hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
      r : Nat
      hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
      ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
      rpos : LT.lt 0 r
      key : Eq r 1
      ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
    -/
  · rw [key, mul_one] at hr
    /-
      case intro.inl
      n : PNat
      K : Type u
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
      ζ : K
      l : Nat
      hζ : IsPrimitiveRoot ζ l
      hl✝ : Ne l 0
      hl : NeZero l
      hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
      r : Nat
      hr : Eq (GCDMonoid.lcm l ↑n) ↑n
      ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
      rpos : LT.lt 0 r
      key : Eq r 1
      ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
    -/
    rw [← hr]
    /-
      case intro.inl
      n : PNat
      K : Type u
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
      ζ : K
      l : Nat
      hζ : IsPrimitiveRoot ζ l
      hl✝ : Ne l 0
      hl : NeZero l
      hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
      r : Nat
      hr : Eq (GCDMonoid.lcm l ↑n) ↑n
      ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
      rpos : LT.lt 0 r
      key : Eq r 1
      ⊢ Dvd.dvd l (HMul.hMul 2 (GCDMonoid.lcm l ↑n))
    -/
    exact dvd_mul_of_dvd_right (_root_.dvd_lcm_left l ↑n) 2
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      n : PNat
      K : Type u
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
      ζ : K
      l : Nat
      hζ : IsPrimitiveRoot ζ l
      hl✝ : Ne l 0
      hl : NeZero l
      hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
      r : Nat
      hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul (↑n) r)
      ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
      rpos : LT.lt 0 r
      key : Eq r 2
      ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
    -/
  · rw [key, mul_comm] at hr
    /-
      case intro.inr
      n : PNat
      K : Type u
      inst✝² : Field K
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
      ζ : K
      l : Nat
      hζ : IsPrimitiveRoot ζ l
      hl✝ : Ne l 0
      hl : NeZero l
      hroot : IsPrimitiveRoot (IsCyclotomicExtension.zeta n Rat K) ↑n
      r : Nat
      hr : Eq (GCDMonoid.lcm l ↑n) (HMul.hMul 2 ↑n)
      ineq : LE.le (HMul.hMul (↑n).totient r.totient) (GCDMonoid.lcm l ↑n).totient
      rpos : LT.lt 0 r
      key : Eq r 2
      ⊢ Dvd.dvd l (HMul.hMul 2 ↑n)
    -/
    simpa [← hr] using _root_.dvd_lcm_left _ _
    /-
      🎉 no goals
    -/


/-- If `x` is a root of unity (spelled as `IsOfFinOrder x`) in an `n`-th cyclotomic extension of
`ℚ`, where `n` is odd, and `ζ` is a primitive `n`-th root of unity, then there exist `r`
such that `x = (-ζ)^r`. -/
theorem exists_neg_pow_of_isOfFinOrder [IsCyclotomicExtension {n} ℚ K]
    (hno : Odd (n : ℕ)) {ζ x : K} (hζ : IsPrimitiveRoot ζ n) (hx : IsOfFinOrder x) :
    ∃ r : ℕ, x = (-ζ) ^ r :=  by
  have hnegζ : IsPrimitiveRoot (-ζ) (2 * n) := by
    convert IsPrimitiveRoot.orderOf (-ζ)
    rw [neg_eq_neg_one_mul, (Commute.all _ _).orderOf_mul_eq_mul_orderOf_of_coprime]
    · simp [hζ.eq_orderOf]
    · simp [← hζ.eq_orderOf, hno]
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  obtain ⟨k, hkpos, hkn⟩ := isOfFinOrder_iff_pow_eq_one.1 hx
  /-
    case intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  obtain ⟨l, hl, hlroot⟩ := (isRoot_of_unity_iff hkpos _).1 hkn
  /-
    case intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlroot : (Polynomial.cyclotomic l K).IsRoot x
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  have hlzero : NeZero l := ⟨fun h ↦ by simp [h] at hl⟩
  /-
    case intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlroot : (Polynomial.cyclotomic l K).IsRoot x
    hlzero : NeZero l
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  have : NeZero (l : K) := ⟨NeZero.natCast_ne l K⟩
  /-
    case intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlroot : (Polynomial.cyclotomic l K).IsRoot x
    hlzero : NeZero l
    this : NeZero ↑l
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  rw [isRoot_cyclotomic_iff] at hlroot
  /-
    case intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlroot : IsPrimitiveRoot x l
    hlzero : NeZero l
    this : NeZero ↑l
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  obtain ⟨a, ha⟩ := hlroot.dvd_of_isCyclotomicExtension n hlzero.1
  /-
    case intro.intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlroot : IsPrimitiveRoot x l
    hlzero : NeZero l
    this : NeZero ↑l
    a : Nat
    ha : Eq (HMul.hMul 2 ↑n) (HMul.hMul l a)
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  replace hlroot : x ^ (2 * (n : ℕ)) = 1 := by rw [ha, pow_mul, hlroot.pow_eq_one, one_pow]
  /-
    case intro.intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlzero : NeZero l
    this : NeZero ↑l
    a : Nat
    ha : Eq (HMul.hMul 2 ↑n) (HMul.hMul l a)
    hlroot : Eq (HPow.hPow x (HMul.hMul 2 ↑n)) 1
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  obtain ⟨s, -, hs⟩ := hnegζ.eq_pow_of_pow_eq_one hlroot
  /-
    case intro.intro.intro.intro.intro.intro.intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    hnegζ : IsPrimitiveRoot (Neg.neg ζ) (HMul.hMul 2 ↑n)
    k : Nat
    hkpos : LT.lt 0 k
    hkn : Eq (HPow.hPow x k) 1
    l : Nat
    hl : Membership.mem k.divisors l
    hlzero : NeZero l
    this : NeZero ↑l
    a : Nat
    ha : Eq (HMul.hMul 2 ↑n) (HMul.hMul l a)
    hlroot : Eq (HPow.hPow x (HMul.hMul 2 ↑n)) 1
    s : Nat
    hs : Eq (HPow.hPow (Neg.neg ζ) s) x
    ⊢ Exists fun r => Eq x (HPow.hPow (Neg.neg ζ) r)
  -/
  exact ⟨s, hs.symm⟩
  /-
    🎉 no goals
  -/


/-- If `x` is a root of unity (spelled as `IsOfFinOrder x`) in an `n`-th cyclotomic extension of
`ℚ`, where `n` is odd, and `ζ` is a primitive `n`-th root of unity, then there exists `r < n`
such that `x = ζ^r` or `x = -ζ^r`. -/
theorem exists_pow_or_neg_mul_pow_of_isOfFinOrder [IsCyclotomicExtension {n} ℚ K]
    (hno : Odd (n : ℕ)) {ζ x : K} (hζ : IsPrimitiveRoot ζ n) (hx : IsOfFinOrder x) :
    ∃ r : ℕ, r < n ∧ (x = ζ ^ r ∨ x = -ζ ^ r) :=  by
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    ⊢ Exists fun r => And (LT.lt r ↑n) (Or (Eq x (HPow.hPow ζ r)) (Eq x (Neg.neg ( …
  -/
  obtain ⟨r, hr⟩ := hζ.exists_neg_pow_of_isOfFinOrder hno hx
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    r : Nat
    hr : Eq x (HPow.hPow (Neg.neg ζ) r)
    ⊢ Exists fun r => And (LT.lt r ↑n) (Or (Eq x (HPow.hPow ζ r)) (Eq x (Neg.neg ( …
  -/
  refine ⟨r % n, Nat.mod_lt _ n.2, ?_⟩
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    r : Nat
    hr : Eq x (HPow.hPow (Neg.neg ζ) r)
    ⊢ Or (Eq x (HPow.hPow ζ (HMod.hMod r ↑n))) (Eq x (Neg.neg (HPow.hPow ζ (HMod.h …
  -/
  rw [show ζ ^ (r % ↑n) = ζ ^ r from (IsPrimitiveRoot.eq_orderOf hζ).symm ▸ pow_mod_orderOf .., hr]
  /-
    case intro
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hno : Odd ↑n
    ζ x : K
    hζ : IsPrimitiveRoot ζ ↑n
    hx : IsOfFinOrder x
    r : Nat
    hr : Eq x (HPow.hPow (Neg.neg ζ) r)
    ⊢ Or (Eq (HPow.hPow (Neg.neg ζ) r) (HPow.hPow ζ r)) (Eq (HPow.hPow (Neg.neg ζ) …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  rcases Nat.even_or_odd r with (h | h) <;> simp [neg_pow, h.neg_one_pow]
                                            /-
                                              🎉 no goals
                                            -/


/-- This mathematically trivial result is complementary to `norm_eq_one` below. -/
theorem norm_eq_neg_one_pow (hζ : IsPrimitiveRoot ζ 2) [IsDomain L] :
    norm K ζ = (-1 : K) ^ finrank K L := by
  /-
    K : Type u
    L : Type v
    inst✝³ : CommRing L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ 2
    inst✝ : IsDomain L
    ⊢ Eq ((Algebra.norm K) ζ) (HPow.hPow (-1) (Module.finrank K L))
  -/
  rw [hζ.eq_neg_one_of_two_right, show -1 = algebraMap K L (-1) by simp, Algebra.norm_algebraMap]
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic n K)` (in particular for `K = ℚ`), the norm of a primitive root is
`1` if `n ≠ 2`. -/
theorem norm_eq_one [IsDomain L] [IsCyclotomicExtension {n} K L] (hn : n ≠ 2)
    (hirr : Irreducible (cyclotomic n K)) : norm K ζ = 1 := by
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : CommRing L
    ζ : L
    inst✝³ : Field K
    inst✝² : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝¹ : IsDomain L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hn : Ne n 2
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq ((Algebra.norm K) ζ) 1
  -/
  haveI := IsCyclotomicExtension.neZero' n K L
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : CommRing L
    ζ : L
    inst✝³ : Field K
    inst✝² : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝¹ : IsDomain L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hn : Ne n 2
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    ⊢ Eq ((Algebra.norm K) ζ) 1
  -/
  by_cases h1 : n = 1
    /-
      case pos
      n : PNat
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      hζ : IsPrimitiveRoot ζ ↑n
      inst✝¹ : IsDomain L
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
      hn : Ne n 2
      hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
      this : NeZero ↑↑n
      h1 : Eq n 1
      ⊢ Eq ((Algebra.norm K) ζ) 1
    -/
  · rw [h1, one_coe, one_right_iff] at hζ
    /-
      case pos
      n : PNat
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      hζ : Eq ζ 1
      inst✝¹ : IsDomain L
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
      hn : Ne n 2
      hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
      this : NeZero ↑↑n
      h1 : Eq n 1
      ⊢ Eq ((Algebra.norm K) ζ) 1
    -/
    rw [hζ, show 1 = algebraMap K L 1 by simp, Algebra.norm_algebraMap, one_pow]
    /-
      🎉 no goals
    -/
  · replace h1 : 2 ≤ n := by
      by_contra! h
      exact h1 (PNat.eq_one_of_lt_two h)
-- Porting note: specifying the type of `cyclotomic_coeff_zero K h1` was not needed.
    rw [← hζ.powerBasis_gen K, PowerBasis.norm_gen_eq_coeff_zero_minpoly, hζ.powerBasis_gen K, ←
      hζ.minpoly_eq_cyclotomic_of_irreducible hirr,
      (cyclotomic_coeff_zero K h1 : coeff (cyclotomic n K) 0 = 1), mul_one,
      hζ.powerBasis_dim K, ← hζ.minpoly_eq_cyclotomic_of_irreducible hirr, natDegree_cyclotomic]
    /-
      case neg
      n : PNat
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      hζ : IsPrimitiveRoot ζ ↑n
      inst✝¹ : IsDomain L
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
      hn : Ne n 2
      hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
      this : NeZero ↑↑n
      h1 : LE.le 2 n
      ⊢ Eq (HPow.hPow (-1) (↑n).totient) 1
    -/
    exact (totient_even <| h1.lt_of_ne hn.symm).neg_one_pow
    /-
      🎉 no goals
    -/


/-- If `K` is linearly ordered, the norm of a primitive root is `1` if `n` is odd. -/
theorem norm_eq_one_of_linearly_ordered {K : Type*} [LinearOrderedField K] [Algebra K L]
    (hodd : Odd (n : ℕ)) : norm K ζ = 1 := by
  /-
    n : PNat
    L : Type v
    inst✝² : CommRing L
    ζ : L
    hζ : IsPrimitiveRoot ζ ↑n
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : Algebra K L
    hodd : Odd ↑n
    ⊢ Eq ((Algebra.norm K) ζ) 1
  -/
  have hz := congr_arg (norm K) ((IsPrimitiveRoot.iff_def _ n).1 hζ).1
  /-
    n : PNat
    L : Type v
    inst✝² : CommRing L
    ζ : L
    hζ : IsPrimitiveRoot ζ ↑n
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : Algebra K L
    hodd : Odd ↑n
    hz : Eq ((Algebra.norm K) (HPow.hPow ζ ↑n)) ((Algebra.norm K) 1)
    ⊢ Eq ((Algebra.norm K) ζ) 1
  -/
  rw [← (algebraMap K L).map_one, Algebra.norm_algebraMap, one_pow, map_pow, ← one_pow ↑n] at hz
  /-
    n : PNat
    L : Type v
    inst✝² : CommRing L
    ζ : L
    hζ : IsPrimitiveRoot ζ ↑n
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : Algebra K L
    hodd : Odd ↑n
    hz : Eq (HPow.hPow ((Algebra.norm K) ζ) ↑n) (HPow.hPow 1 ↑n)
    ⊢ Eq ((Algebra.norm K) ζ) 1
  -/
  exact StrictMono.injective hodd.strictMono_pow hz
  /-
    🎉 no goals
  -/


theorem norm_of_cyclotomic_irreducible [IsDomain L] [IsCyclotomicExtension {n} K L]
    (hirr : Irreducible (cyclotomic n K)) : norm K ζ = ite (n = 2) (-1) 1 := by
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝⁴ : CommRing L
    ζ : L
    inst✝³ : Field K
    inst✝² : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝¹ : IsDomain L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq ((Algebra.norm K) ζ) (ite (Eq n 2) (-1) 1)
  -/
  split_ifs with hn
    /-
      case pos
      n : PNat
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      hζ : IsPrimitiveRoot ζ ↑n
      inst✝¹ : IsDomain L
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
      hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
      hn : Eq n 2
      ⊢ Eq ((Algebra.norm K) ζ) (-1)
    -/
  · subst hn
    /-
      case pos
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      inst✝¹ : IsDomain L
      hζ : IsPrimitiveRoot ζ ↑2
      inst✝ : IsCyclotomicExtension (Singleton.singleton 2) K L
      hirr : Irreducible (Polynomial.cyclotomic (↑2) K)
      ⊢ Eq ((Algebra.norm K) ζ) (-1)
    -/
    convert norm_eq_neg_one_pow (K := K) hζ
    /-
      case h.e'_3
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      inst✝¹ : IsDomain L
      hζ : IsPrimitiveRoot ζ ↑2
      inst✝ : IsCyclotomicExtension (Singleton.singleton 2) K L
      hirr : Irreducible (Polynomial.cyclotomic (↑2) K)
      ⊢ Eq (-1) (HPow.hPow (-1) (Module.finrank K L))
    -/
    erw [IsCyclotomicExtension.finrank _ hirr, totient_two, pow_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : PNat
      K : Type u
      L : Type v
      inst✝⁴ : CommRing L
      ζ : L
      inst✝³ : Field K
      inst✝² : Algebra K L
      hζ : IsPrimitiveRoot ζ ↑n
      inst✝¹ : IsDomain L
      inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
      hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
      hn : Not (Eq n 2)
      ⊢ Eq ((Algebra.norm K) ζ) 1
    -/
  · exact hζ.norm_eq_one hn hirr
    /-
      🎉 no goals
    -/


/-- If `Irreducible (cyclotomic n K)` (in particular for `K = ℚ`), then the norm of
`ζ - 1` is `eval 1 (cyclotomic n ℤ)`. -/
theorem sub_one_norm_eq_eval_cyclotomic [IsCyclotomicExtension {n} K L] (h : 2 < (n : ℕ))
    (hirr : Irreducible (cyclotomic n K)) : norm K (ζ - 1) = ↑(eval 1 (cyclotomic n ℤ)) := by
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(Polynomial.eval 1 (Polynomial.cyclot …
  -/
  haveI := IsCyclotomicExtension.neZero' n K L
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(Polynomial.eval 1 (Polynomial.cyclot …
  -/
  let E := AlgebraicClosure L
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(Polynomial.eval 1 (Polynomial.cyclot …
  -/
  obtain ⟨z, hz⟩ := IsAlgClosed.exists_root _ (degree_cyclotomic_pos n E n.pos).ne.symm
  /-
    case intro
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    z : E
    hz : (Polynomial.cyclotomic (↑n) E).IsRoot z
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(Polynomial.eval 1 (Polynomial.cyclot …
  -/
  apply (algebraMap K E).injective
  /-
    case intro.a
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    z : E
    hz : (Polynomial.cyclotomic (↑n) E).IsRoot z
    ⊢ Eq ((algebraMap K E) ((Algebra.norm K) (HSub.hSub ζ 1))) ((algebraMap K E) ↑ …
  -/
  letI := IsCyclotomicExtension.finiteDimensional {n} K L
  /-
    case intro.a
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this✝ : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    z : E
    hz : (Polynomial.cyclotomic (↑n) E).IsRoot z
    this : FiniteDimensional K L := IsCyclotomicExtension.finiteDimensional (Singl …
    ⊢ Eq ((algebraMap K E) ((Algebra.norm K) (HSub.hSub ζ 1))) ((algebraMap K E) ↑ …
  -/
  letI := IsCyclotomicExtension.isGalois n K L
  /-
    case intro.a
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this✝¹ : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    z : E
    hz : (Polynomial.cyclotomic (↑n) E).IsRoot z
    this✝ : FiniteDimensional K L := IsCyclotomicExtension.finiteDimensional (Sing …
    this : IsGalois K L := IsCyclotomicExtension.isGalois n K L
    ⊢ Eq ((algebraMap K E) ((Algebra.norm K) (HSub.hSub ζ 1))) ((algebraMap K E) ↑ …
  -/
  rw [norm_eq_prod_embeddings]
  conv_lhs =>
    congr
    rfl
    ext
    rw [← neg_sub, map_neg, map_sub, map_one, neg_eq_neg_one_mul]
  rw [prod_mul_distrib, prod_const, Finset.card_univ, AlgHom.card,
    IsCyclotomicExtension.finrank L hirr, (totient_even h).neg_one_pow, one_mul]
  have Hprod : (Finset.univ.prod fun σ : L →ₐ[K] E => 1 - σ ζ) = eval 1 (cyclotomic' n E) := by
    rw [cyclotomic', eval_prod, ← @Finset.prod_attach E E, ← univ_eq_attach]
    refine Fintype.prod_equiv (hζ.embeddingsEquivPrimitiveRoots E hirr) _ _ fun σ => ?_
    simp
  /-
    case intro.a
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : LT.lt 2 ↑n
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    this✝¹ : NeZero ↑↑n
    E : Type v := AlgebraicClosure L
    z : E
    hz : (Polynomial.cyclotomic (↑n) E).IsRoot z
    this✝ : FiniteDimensional K L := IsCyclotomicExtension.finiteDimensional (Sing …
    this : IsGalois K L := IsCyclotomicExtension.isGalois n K L
    Hprod : Eq (Finset.univ.prod fun σ => HSub.hSub 1 (σ ζ)) (Polynomial.eval 1 (P …
    ⊢ Eq (Finset.univ.prod fun x => HSub.hSub 1 (x ζ)) ((algebraMap K E) ↑(Polynom …
  -/
  haveI : NeZero ((n : ℕ) : E) := NeZero.of_noZeroSMulDivisors K _ (n : ℕ)
  rw [Hprod, cyclotomic', ← cyclotomic_eq_prod_X_sub_primitiveRoots (isRoot_cyclotomic_iff.1 hz),
    ← map_cyclotomic_int, _root_.map_intCast, ← Int.cast_one, eval_intCast_map, eq_intCast,
    Int.cast_id]


/-- If `IsPrimePow (n : ℕ)`, `n ≠ 2` and `Irreducible (cyclotomic n K)` (in particular for
`K = ℚ`), then the norm of `ζ - 1` is `(n : ℕ).minFac`. -/
theorem sub_one_norm_isPrimePow (hn : IsPrimePow (n : ℕ)) [IsCyclotomicExtension {n} K L]
    (hirr : Irreducible (cyclotomic (n : ℕ) K)) (h : n ≠ 2) : norm K (ζ - 1) = (n : ℕ).minFac := by
  have :=
    (coe_lt_coe 2 _).1
      (lt_of_le_of_ne (succ_le_of_lt (IsPrimePow.one_lt hn))
        (Function.Injective.ne PNat.coe_injective h).symm)
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    hn : IsPrimePow ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    h : Ne n 2
    this : LT.lt 2 n
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(↑n).minFac
  -/
  letI hprime : Fact (n : ℕ).minFac.Prime := ⟨minFac_prime (IsPrimePow.ne_one hn)⟩
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    hn : IsPrimePow ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    h : Ne n 2
    this : LT.lt 2 n
    hprime : Fact (Nat.Prime (↑n).minFac) := { out := Nat.minFac_prime (IsPrimePow …
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑(↑n).minFac
  -/
  rw [sub_one_norm_eq_eval_cyclotomic hζ this hirr]
  /-
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    hn : IsPrimePow ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    h : Ne n 2
    this : LT.lt 2 n
    hprime : Fact (Nat.Prime (↑n).minFac) := { out := Nat.minFac_prime (IsPrimePow …
    ⊢ Eq ↑(Polynomial.eval 1 (Polynomial.cyclotomic (↑n) Int)) ↑(↑n).minFac
  -/
  nth_rw 1 [← IsPrimePow.minFac_pow_factorization_eq hn]
  obtain ⟨k, hk⟩ : ∃ k, (n : ℕ).factorization (n : ℕ).minFac = k + 1 :=
    exists_eq_succ_of_ne_zero
      (((n : ℕ).factorization.mem_support_toFun (n : ℕ).minFac).1 <|
        mem_primeFactors_iff_mem_primeFactorsList.2 <|
          (mem_primeFactorsList (IsPrimePow.ne_zero hn)).2 ⟨hprime.out, minFac_dvd _⟩)
  /-
    case intro
    n : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    hζ : IsPrimitiveRoot ζ ↑n
    hn : IsPrimePow ↑n
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    hirr : Irreducible (Polynomial.cyclotomic (↑n) K)
    h : Ne n 2
    this : LT.lt 2 n
    hprime : Fact (Nat.Prime (↑n).minFac) := { out := Nat.minFac_prime (IsPrimePow …
    k : Nat
    hk : Eq ((↑n).factorization (↑n).minFac) (HAdd.hAdd k 1)
    ⊢ Eq ↑(Polynomial.eval 1 (Polynomial.cyclotomic (HPow.hPow (↑n).minFac ((↑n).f …
  -/
  simp [hk, sub_one_norm_eq_eval_cyclotomic hζ this hirr]
  /-
    🎉 no goals
  -/


theorem minpoly_sub_one_eq_cyclotomic_comp [Algebra K A] [IsDomain A] {ζ : A}
    [IsCyclotomicExtension {n} K A] (hζ : IsPrimitiveRoot ζ n)
    (h : Irreducible (Polynomial.cyclotomic n K)) :
    minpoly K (ζ - 1) = (cyclotomic n K).comp (X + 1) := by
  /-
    n : PNat
    A : Type w
    K : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra K A
    inst✝¹ : IsDomain A
    ζ : A
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K A
    hζ : IsPrimitiveRoot ζ ↑n
    h : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq (minpoly K (HSub.hSub ζ 1)) ((Polynomial.cyclotomic (↑n) K).comp (HAdd.hA …
  -/
  haveI := IsCyclotomicExtension.neZero' n K A
  rw [show ζ - 1 = ζ + algebraMap K A (-1) by simp [sub_eq_add_neg],
    minpoly.add_algebraMap ζ,
    hζ.minpoly_eq_cyclotomic_of_irreducible h]
  /-
    n : PNat
    A : Type w
    K : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra K A
    inst✝¹ : IsDomain A
    ζ : A
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K A
    hζ : IsPrimitiveRoot ζ ↑n
    h : Irreducible (Polynomial.cyclotomic (↑n) K)
    this : NeZero ↑↑n
    ⊢ Eq ((minpoly K ζ).comp (HSub.hSub Polynomial.X (Polynomial.C (-1)))) ((minpo …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is a prime,
then the norm of `ζ ^ (p ^ s) - 1` is `p ^ (p ^ s)` if `p ^ (k - s + 1) ≠ 2`. See the next lemmas
for similar results. -/
theorem norm_pow_sub_one_of_prime_pow_ne_two {k s : ℕ} (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1)))
    [hpri : Fact (p : ℕ).Prime] [IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) (hs : s ≤ k)
    (htwo : p ^ (k - s + 1) ≠ 2) : norm K (ζ ^ (p : ℕ) ^ s - 1) = (p : K) ^ (p : ℕ) ^ s := by
  have hirr₁ : Irreducible (cyclotomic ((p : ℕ) ^ (k - s + 1)) K) :=
    cyclotomic_irreducible_pow_of_irreducible_pow hpri.1 (by omega) hirr
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (HPow.hPow (↑p) (HAdd.hAdd (HSub.hS …
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1)) (HPow.h …
  -/
  rw [← PNat.pow_coe] at hirr₁
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1)) (HPow.h …
  -/
  set η := ζ ^ (p : ℕ) ^ s - 1
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    ⊢ Eq ((Algebra.norm K) η) (HPow.hPow (↑↑p) (HPow.hPow (↑p) s))
  -/
  let η₁ : K⟮η⟯ := IntermediateField.AdjoinSimple.gen K η
  have hη : IsPrimitiveRoot (η + 1) ((p : ℕ) ^ (k + 1 - s)) := by
    rw [sub_add_cancel]
    refine IsPrimitiveRoot.pow (p ^ (k + 1)).pos hζ ?_
    rw [PNat.pow_coe, ← pow_add, add_comm s, Nat.sub_add_cancel (le_trans hs (Nat.le_succ k))]
  have : IsCyclotomicExtension {p ^ (k - s + 1)} K K⟮η⟯ := by
    have HKη : K⟮η⟯ = K⟮η + 1⟯ := by
      refine le_antisymm ?_ ?_
      all_goals rw [IntermediateField.adjoin_simple_le_iff]
      · nth_rw 2 [← add_sub_cancel_right η 1]
        exact sub_mem (IntermediateField.mem_adjoin_simple_self K (η + 1)) (one_mem _)
      · exact add_mem (IntermediateField.mem_adjoin_simple_self K η) (one_mem _)
    rw [HKη]
    have H := IntermediateField.adjoin_simple_toSubalgebra_of_integral
      ((integral {p ^ (k + 1)} K L).isIntegral (η + 1))
    refine IsCyclotomicExtension.equiv _ _ _ (h := ?_) (.refl : K⟮η + 1⟯.toSubalgebra ≃ₐ[K] _)
    rw [H]
    have hη' : IsPrimitiveRoot (η + 1) ↑(p ^ (k + 1 - s)) := by simpa using hη
-- Porting note: `using 1` was not needed.
    convert hη'.adjoin_isCyclotomicExtension K using 1
    rw [Nat.sub_add_comm hs]
  replace hη : IsPrimitiveRoot (η₁ + 1) ↑(p ^ (k - s + 1)) := by
    apply coe_submonoidClass_iff.1
    convert hη using 1
    rw [Nat.sub_add_comm hs, pow_coe]
-- Porting note: the following `have` were not needed because the locale `cyclotomic` set them
-- as instances.
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (HSu …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    ⊢ Eq ((Algebra.norm K) η) (HPow.hPow (↑↑p) (HPow.hPow (↑p) s))
  -/
  have := IsCyclotomicExtension.finiteDimensional {p ^ (k + 1)} K L
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (HS …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this : FiniteDimensional K L
    ⊢ Eq ((Algebra.norm K) η) (HPow.hPow (↑↑p) (HPow.hPow (↑p) s))
  -/
  have := IsCyclotomicExtension.isGalois (p ^ (k + 1)) K L
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    ⊢ Eq ((Algebra.norm K) η) (HPow.hPow (↑↑p) (HPow.hPow (↑p) s))
  -/
  rw [norm_eq_norm_adjoin K]
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    ⊢ Eq (HPow.hPow ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K η)) (M …
  -/
  have H := hη.sub_one_norm_isPrimePow ?_ hirr₁ htwo
  /-
    case refine_2
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    H : Eq ((Algebra.norm K) (HSub.hSub (HAdd.hAdd η₁ 1) 1)) ↑(↑(HPow.hPow p (HAdd …
    ⊢ Eq (HPow.hPow ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K η)) (M …
  -/
  swap; · rw [PNat.pow_coe]; exact hpri.1.isPrimePow.pow (Nat.succ_ne_zero _)
                             /-
                               🎉 no goals
                             -/
  /-
    case refine_2
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    H : Eq ((Algebra.norm K) (HSub.hSub (HAdd.hAdd η₁ 1) 1)) ↑(↑(HPow.hPow p (HAdd …
    ⊢ Eq (HPow.hPow ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K η)) (M …
  -/
  rw [add_sub_cancel_right] at H
  /-
    case refine_2
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    ⊢ Eq (HPow.hPow ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K η)) (M …
  -/
  rw [H]
  /-
    case refine_2
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    ⊢ Eq (HPow.hPow (↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).minFac) (Modu …
  -/
  congr
    /-
      case refine_2.e_a.e_a
      p : PNat
      K : Type u
      L : Type v
      inst✝³ : Field L
      ζ : L
      inst✝² : Field K
      inst✝¹ : Algebra K L
      k s : Nat
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hpri : Fact (Nat.Prime ↑p)
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hs : LE.le s k
      htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
      hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
      η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
      η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
      this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
      hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
      this✝ : FiniteDimensional K L
      this : IsGalois K L
      H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
      ⊢ Eq (↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).minFac ↑p
    -/
  · rw [PNat.pow_coe, Nat.pow_minFac, hpri.1.minFac_eq]
    /-
      case refine_2.e_a.e_a
      p : PNat
      K : Type u
      L : Type v
      inst✝³ : Field L
      ζ : L
      inst✝² : Field K
      inst✝¹ : Algebra K L
      k s : Nat
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hpri : Fact (Nat.Prime ↑p)
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hs : LE.le s k
      htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
      hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
      η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
      η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
      this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
      hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
      this✝ : FiniteDimensional K L
      this : IsGalois K L
      H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
      ⊢ Ne (HAdd.hAdd (HSub.hSub k s) 1) 0
    -/
    exact Nat.succ_ne_zero _
    /-
      🎉 no goals
    -/
  /-
    case refine_2.e_a
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝¹ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝ : FiniteDimensional K L
    this : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField.adjoi …
  -/
  have := Module.finrank_mul_finrank K K⟮η⟯ L
  rw [IsCyclotomicExtension.finrank L hirr, IsCyclotomicExtension.finrank K⟮η⟯ hirr₁,
    PNat.pow_coe, PNat.pow_coe, Nat.totient_prime_pow hpri.out (k - s).succ_pos,
    Nat.totient_prime_pow hpri.out k.succ_pos, mul_comm _ ((p : ℕ) - 1), mul_assoc,
    mul_comm ((p : ℕ) ^ (k.succ - 1))] at this
  /-
    case refine_2.e_a
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝² : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝¹ : FiniteDimensional K L
    this✝ : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    this : Eq (HMul.hMul (HSub.hSub (↑p) 1) (HMul.hMul (HPow.hPow (↑p) (HSub.hSub  …
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField.adjoi …
  -/
  replace this := mul_left_cancel₀ (tsub_pos_iff_lt.2 hpri.out.one_lt).ne' this
  have Hex : k.succ - 1 = (k - s).succ - 1 + s := by
    simp only [Nat.succ_sub_succ_eq_sub, tsub_zero]
    exact (Nat.sub_add_cancel hs).symm
  /-
    case refine_2.e_a
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝² : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝¹ : FiniteDimensional K L
    this✝ : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    this : Eq (HMul.hMul (HPow.hPow (↑p) (HSub.hSub (HSub.hSub k s).succ 1)) (Modu …
    Hex : Eq (HSub.hSub k.succ 1) (HAdd.hAdd (HSub.hSub (HSub.hSub k s).succ 1) s)
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField.adjoi …
  -/
  rw [Hex, pow_add] at this
  /-
    case refine_2.e_a
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    hirr₁ : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd (HSub.hSu …
    η : L := HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1
    η₁ : Subtype fun x => Membership.mem (IntermediateField.adjoin K (Singleton.si …
    this✝² : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd (H …
    hη : IsPrimitiveRoot (HAdd.hAdd η₁ 1) ↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) …
    this✝¹ : FiniteDimensional K L
    this✝ : IsGalois K L
    H : Eq ((Algebra.norm K) η₁) ↑(↑(HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1))).m …
    this : Eq (HMul.hMul (HPow.hPow (↑p) (HSub.hSub (HSub.hSub k s).succ 1)) (Modu …
    Hex : Eq (HSub.hSub k.succ 1) (HAdd.hAdd (HSub.hSub (HSub.hSub k s).succ 1) s)
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField.adjoi …
  -/
  exact mul_left_cancel₀ (pow_ne_zero _ hpri.out.ne_zero) this
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is a prime,
then the norm of `ζ ^ (p ^ s) - 1` is `p ^ (p ^ s)` if `p ≠ 2`. -/
theorem norm_pow_sub_one_of_prime_ne_two {k : ℕ} (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1)))
    [hpri : Fact (p : ℕ).Prime] [IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) {s : ℕ} (hs : s ≤ k) (hodd : p ≠ 2) :
    norm K (ζ ^ (p : ℕ) ^ s - 1) = (p : K) ^ (p : ℕ) ^ s := by
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1)) (HPow.h …
  -/
  refine hζ.norm_pow_sub_one_of_prime_pow_ne_two hirr hs fun h => ?_
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ False
  -/
  have coe_two : ((2 : ℕ+) : ℕ) = 2 := by norm_cast
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    coe_two : Eq (↑2) 2
    ⊢ False
  -/
  rw [← PNat.coe_inj, coe_two, PNat.pow_coe, ← pow_one 2] at h
-- Porting note: the proof is slightly different because of coercions.
  replace h :=
    eq_of_prime_pow_eq (prime_iff.1 hpri.out) (prime_iff.1 Nat.prime_two) (k - s).succ_pos h
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    coe_two : Eq (↑2) 2
    h : Eq (↑p) 2
    ⊢ False
  -/
  exact hodd (PNat.coe_injective h)
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is an odd
prime, then the norm of `ζ - 1` is `p`. -/
theorem norm_sub_one_of_prime_ne_two {k : ℕ} (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1)))
    [hpri : Fact (p : ℕ).Prime] [IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) (h : p ≠ 2) : norm K (ζ - 1) = p := by
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    h : Ne p 2
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑↑p
  -/
  simpa using hζ.norm_pow_sub_one_of_prime_ne_two hirr k.zero_le h
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic p K)` (in particular for `K = ℚ`) and `p` is an odd prime,
then the norm of `ζ - 1` is `p`. -/
theorem norm_sub_one_of_prime_ne_two' [hpri : Fact (p : ℕ).Prime]
    [hcyc : IsCyclotomicExtension {p} K L] (hζ : IsPrimitiveRoot ζ p)
    (hirr : Irreducible (cyclotomic p K)) (h : p ≠ 2) : norm K (ζ - 1) = p := by
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    hpri : Fact (Nat.Prime ↑p)
    hcyc : IsCyclotomicExtension (Singleton.singleton p) K L
    hζ : IsPrimitiveRoot ζ ↑p
    hirr : Irreducible (Polynomial.cyclotomic (↑p) K)
    h : Ne p 2
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑↑p
  -/
  replace hirr : Irreducible (cyclotomic (p ^ (0 + 1) : ℕ) K) := by simp [hirr]
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    hpri : Fact (Nat.Prime ↑p)
    hcyc : IsCyclotomicExtension (Singleton.singleton p) K L
    hζ : IsPrimitiveRoot ζ ↑p
    h : Ne p 2
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow (↑p) (HAdd.hAdd 0 1)) K)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑↑p
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1) : ℕ) := by simp [hζ]
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    hpri : Fact (Nat.Prime ↑p)
    hcyc : IsCyclotomicExtension (Singleton.singleton p) K L
    h : Ne p 2
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow (↑p) (HAdd.hAdd 0 1)) K)
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑↑p
  -/
  haveI : IsCyclotomicExtension {p ^ (0 + 1)} K L := by simp [hcyc]
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    hpri : Fact (Nat.Prime ↑p)
    hcyc : IsCyclotomicExtension (Singleton.singleton p) K L
    h : Ne p 2
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow (↑p) (HAdd.hAdd 0 1)) K)
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) ↑↑p
  -/
  simpa using norm_sub_one_of_prime_ne_two hζ hirr h
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (2 ^ (k + 1)) K)` (in particular for `K = ℚ`), then the norm of
`ζ ^ (2 ^ k) - 1` is `(-2) ^ (2 ^ k)`. -/
-- Porting note: writing `(2 : ℕ+)` was not needed (similarly everywhere).
theorem norm_pow_sub_one_two {k : ℕ} (hζ : IsPrimitiveRoot ζ (2 ^ (k + 1)))
    [IsCyclotomicExtension {(2 : ℕ+) ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (2 ^ (k + 1)) K)) :
    norm K (ζ ^ 2 ^ k - 1) = (-2 : K) ^ 2 ^ k := by
  /-
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 (HAdd.hAdd k 1)) K)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow 2 k)) 1)) (HPow.hPow …
  -/
  have := hζ.pow_of_dvd (fun h => two_ne_zero (pow_eq_zero h)) (pow_dvd_pow 2 (le_succ k))
  /-
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 (HAdd.hAdd k 1)) K)
    this : IsPrimitiveRoot (HPow.hPow ζ (HPow.hPow 2 k)) (HDiv.hDiv (HPow.hPow 2 ( …
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow 2 k)) 1)) (HPow.hPow …
  -/
  rw [Nat.pow_div (le_succ k) zero_lt_two, Nat.succ_sub (le_refl k), Nat.sub_self, pow_one] at this
  have H : (-1 : L) - (1 : L) = algebraMap K L (-2) := by
    simp only [map_neg, map_ofNat]
    ring
  /-
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 (HAdd.hAdd k 1)) K)
    this : IsPrimitiveRoot (HPow.hPow ζ (HPow.hPow 2 k)) 2
    H : Eq (HSub.hSub (-1) 1) ((algebraMap K L) (-2))
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow 2 k)) 1)) (HPow.hPow …
  -/
  replace hirr : Irreducible (cyclotomic ((2 : ℕ+) ^ (k + 1) : ℕ+) K) := by simp [hirr]
-- Porting note: the proof is slightly different because of coercions.
  rw [this.eq_neg_one_of_two_right, H, Algebra.norm_algebraMap,
    IsCyclotomicExtension.finrank L hirr, pow_coe, show ((2 : ℕ+) : ℕ) = 2 from rfl,
      totient_prime_pow Nat.prime_two (zero_lt_succ k), succ_sub_succ_eq_sub, tsub_zero]
  /-
    K : Type u
    L : Type v
    inst✝³ : Field L
    ζ : L
    inst✝² : Field K
    inst✝¹ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    this : IsPrimitiveRoot (HPow.hPow ζ (HPow.hPow 2 k)) 2
    H : Eq (HSub.hSub (-1) 1) ((algebraMap K L) (-2))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
    ⊢ Eq (HPow.hPow (-2) (HMul.hMul (HPow.hPow 2 k) (HSub.hSub 2 1))) (HPow.hPow ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (2 ^ k) K)` (in particular for `K = ℚ`) and `k` is at least `2`,
then the norm of `ζ - 1` is `2`. -/
theorem norm_sub_one_two {k : ℕ} (hζ : IsPrimitiveRoot ζ (2 ^ k)) (hk : 2 ≤ k)
    [H : IsCyclotomicExtension {(2 : ℕ+) ^ k} K L] (hirr : Irreducible (cyclotomic (2 ^ k) K)) :
    norm K (ζ - 1) = 2 := by
  have : 2 < (2 : ℕ+) ^ k := by
    simp only [← coe_lt_coe, one_coe, pow_coe]
    nth_rw 1 [← pow_one 2]
    exact Nat.pow_lt_pow_right one_lt_two (lt_of_lt_of_le one_lt_two hk)
  /-
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 k)
    hk : LE.le 2 k
    H : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 k)) K L
    hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 k) K)
    this : LT.lt 2 (HPow.hPow 2 k)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) 2
  -/
  replace hirr : Irreducible (cyclotomic ((2 : ℕ+) ^ k : ℕ+) K) := by simp [hirr]
  /-
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    k : Nat
    hζ : IsPrimitiveRoot ζ (HPow.hPow 2 k)
    hk : LE.le 2 k
    H : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 k)) K L
    this : LT.lt 2 (HPow.hPow 2 k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 k)) K)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) 2
  -/
  replace hζ : IsPrimitiveRoot ζ (2 ^ k : ℕ+) := by simp [hζ]
  /-
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    k : Nat
    hk : LE.le 2 k
    H : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 k)) K L
    this : LT.lt 2 (HPow.hPow 2 k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 k)) K)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 k)
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) 2
  -/
  obtain ⟨k₁, hk₁⟩ := exists_eq_succ_of_ne_zero (lt_of_lt_of_le zero_lt_two hk).ne.symm
-- Porting note: the proof is slightly different because of coercions.
  /-
    case intro
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    k : Nat
    hk : LE.le 2 k
    H : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 k)) K L
    this : LT.lt 2 (HPow.hPow 2 k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 k)) K)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 k)
    k₁ : Nat
    hk₁ : Eq k k₁.succ
    ⊢ Eq ((Algebra.norm K) (HSub.hSub ζ 1)) 2
  -/
  simpa [hk₁, show ((2 : ℕ+) : ℕ) = 2 from rfl] using sub_one_norm_eq_eval_cyclotomic hζ this hirr
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is a prime,
then the norm of `ζ ^ (p ^ s) - 1` is `p ^ (p ^ s)` if `k ≠ 0` and `s ≤ k`. -/
theorem norm_pow_sub_one_eq_prime_pow_of_ne_zero {k s : ℕ} (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1)))
    [hpri : Fact (p : ℕ).Prime] [hcycl : IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) (hs : s ≤ k) (hk : k ≠ 0) :
    norm K (ζ ^ (p : ℕ) ^ s - 1) = (p : K) ^ (p : ℕ) ^ s := by
  /-
    p : PNat
    K : Type u
    L : Type v
    inst✝² : Field L
    ζ : L
    inst✝¹ : Field K
    inst✝ : Algebra K L
    k s : Nat
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hpri : Fact (Nat.Prime ↑p)
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hs : LE.le s k
    hk : Ne k 0
    ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1)) (HPow.h …
  -/
  by_cases htwo : p ^ (k - s + 1) = 2
  · have hp : p = 2 := by
      rw [← PNat.coe_inj, PNat.pow_coe, ← pow_one 2] at htwo
      replace htwo :=
        eq_of_prime_pow_eq (prime_iff.1 hpri.out) (prime_iff.1 Nat.prime_two) (succ_pos _) htwo
      rwa [show 2 = ((2 : ℕ+) : ℕ) by decide, PNat.coe_inj] at htwo
    replace hs : s = k := by
      rw [hp, ← PNat.coe_inj, PNat.pow_coe] at htwo
      nth_rw 2 [← pow_one 2] at htwo
      replace htwo := Nat.pow_right_injective rfl.le htwo
      rw [add_left_eq_self, Nat.sub_eq_zero_iff_le] at htwo
      exact le_antisymm hs htwo
    simp only [hs, hp, one_coe, cast_one, pow_coe, show ((2 : ℕ+) : ℕ) = 2 from rfl]
      at hζ hirr hcycl ⊢
    /-
      case pos
      p : PNat
      K : Type u
      L : Type v
      inst✝² : Field L
      ζ : L
      inst✝¹ : Field K
      inst✝ : Algebra K L
      k s : Nat
      hpri : Fact (Nat.Prime ↑p)
      hk : Ne k 0
      htwo : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
      hp : Eq p 2
      hs : Eq s k
      hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 (HAdd.hAdd k 1)) K)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow 2 k)) 1)) (HPow.hPow …
    -/
    obtain ⟨k₁, hk₁⟩ := Nat.exists_eq_succ_of_ne_zero hk
-- Porting note: the proof is slightly different because of coercions.
    rw [hζ.norm_pow_sub_one_two hirr, hk₁, _root_.pow_succ', pow_mul, neg_eq_neg_one_mul,
      mul_pow, neg_one_sq, one_mul, ← pow_mul, ← _root_.pow_succ']
    /-
      case pos.intro
      p : PNat
      K : Type u
      L : Type v
      inst✝² : Field L
      ζ : L
      inst✝¹ : Field K
      inst✝ : Algebra K L
      k s : Nat
      hpri : Fact (Nat.Prime ↑p)
      hk : Ne k 0
      htwo : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
      hp : Eq p 2
      hs : Eq s k
      hζ : IsPrimitiveRoot ζ (HPow.hPow 2 (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (HPow.hPow 2 (HAdd.hAdd k 1)) K)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      k₁ : Nat
      hk₁ : Eq k k₁.succ
      ⊢ Eq (HPow.hPow 2 (HPow.hPow 2 (HAdd.hAdd k₁ 1))) (HPow.hPow (↑2) (HPow.hPow 2 …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : PNat
      K : Type u
      L : Type v
      inst✝² : Field L
      ζ : L
      inst✝¹ : Field K
      inst✝ : Algebra K L
      k s : Nat
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hpri : Fact (Nat.Prime ↑p)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hs : LE.le s k
      hk : Ne k 0
      htwo : Not (Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2)
      ⊢ Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) s)) 1)) (HPow.h …
    -/
  · exact hζ.norm_pow_sub_one_of_prime_pow_ne_two hirr hs htwo
    /-
      🎉 no goals
    -/


/-- If `Irreducible (cyclotomic n K)` (in particular for `K = ℚ`), the norm of `zeta n K L` is `1`
if `n` is odd. -/
theorem norm_zeta_eq_one [IsCyclotomicExtension {n} K L] (hn : n ≠ 2)
    (hirr : Irreducible (cyclotomic n K)) : norm K (zeta n K L) = 1 :=
  (zeta_spec n K L).norm_eq_one hn hirr


/-- If `IsPrimePow (n : ℕ)`, `n ≠ 2` and `Irreducible (cyclotomic n K)` (in particular for
`K = ℚ`), then the norm of `zeta n K L - 1` is `(n : ℕ).minFac`. -/
theorem norm_zeta_sub_one_of_isPrimePow (hn : IsPrimePow (n : ℕ)) [IsCyclotomicExtension {n} K L]
    (hirr : Irreducible (cyclotomic (n : ℕ) K)) (h : n ≠ 2) :
    norm K (zeta n K L - 1) = (n : ℕ).minFac :=
  (zeta_spec n K L).sub_one_norm_isPrimePow hn hirr h


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is a prime,
then the norm of `(zeta (p ^ (k + 1)) K L) ^ (p ^ s) - 1` is `p ^ (p ^ s)`
if `p ^ (k - s + 1) ≠ 2`. -/
theorem norm_zeta_pow_sub_one_of_prime_pow_ne_two {k : ℕ} [Fact (p : ℕ).Prime]
    [IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) {s : ℕ} (hs : s ≤ k)
    (htwo : p ^ (k - s + 1) ≠ 2) :
    norm K (zeta (p ^ (k + 1)) K L ^ (p : ℕ) ^ s - 1) = (p : K) ^ (p : ℕ) ^ s :=
  (zeta_spec _ K L).norm_pow_sub_one_of_prime_pow_ne_two hirr hs htwo


/-- If `Irreducible (cyclotomic (p ^ (k + 1)) K)` (in particular for `K = ℚ`) and `p` is an odd
prime, then the norm of `zeta (p ^ (k + 1)) K L - 1` is `p`. -/
theorem norm_zeta_pow_sub_one_of_prime_ne_two {k : ℕ} [Fact (p : ℕ).Prime]
    [IsCyclotomicExtension {p ^ (k + 1)} K L]
    (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K)) (h : p ≠ 2) :
    norm K (zeta (p ^ (k + 1)) K L - 1) = p :=
  (zeta_spec _ K L).norm_sub_one_of_prime_ne_two hirr h


/-- If `Irreducible (cyclotomic p K)` (in particular for `K = ℚ`) and `p` is an odd prime,
then the norm of `zeta p K L - 1` is `p`. -/
theorem norm_zeta_sub_one_of_prime_ne_two [Fact (p : ℕ).Prime]
    [IsCyclotomicExtension {p} K L] (hirr : Irreducible (cyclotomic p K)) (h : p ≠ 2) :
    norm K (zeta p K L - 1) = p :=
  (zeta_spec _ K L).norm_sub_one_of_prime_ne_two' hirr h


/-- If `Irreducible (cyclotomic (2 ^ k) K)` (in particular for `K = ℚ`) and `k` is at least `2`,
then the norm of `zeta (2 ^ k) K L - 1` is `2`. -/
theorem norm_zeta_pow_sub_one_two {k : ℕ} (hk : 2 ≤ k)
    [IsCyclotomicExtension {(2 : ℕ+) ^ k} K L] (hirr : Irreducible (cyclotomic (2 ^ k) K)) :
    norm K (zeta ((2 : ℕ+) ^ k) K L - 1) = 2 :=
  norm_sub_one_two (zeta_spec ((2 : ℕ+) ^ k) K L) hk hirr


@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.pow_sub_one_norm_prime_pow_ne_two :=
  IsPrimitiveRoot.norm_pow_sub_one_of_prime_pow_ne_two

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.pow_sub_one_norm_prime_ne_two :=
  IsPrimitiveRoot.norm_pow_sub_one_of_prime_ne_two

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.sub_one_norm_prime_ne_two :=
  IsPrimitiveRoot.norm_sub_one_of_prime_ne_two

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.sub_one_norm_prime :=
  IsPrimitiveRoot.norm_sub_one_of_prime_ne_two'

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.pow_sub_one_norm_two :=
  IsPrimitiveRoot.norm_pow_sub_one_two

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.sub_one_norm_two :=
  IsPrimitiveRoot.norm_sub_one_two

@[deprecated (since := "2024-04-02")] alias IsPrimitiveRoot.pow_sub_one_norm_prime_pow_of_ne_zero :=
  IsPrimitiveRoot.norm_pow_sub_one_eq_prime_pow_of_ne_zero

@[deprecated (since := "2024-04-02")] alias IsCyclotomicExtension.isPrimePow_norm_zeta_sub_one :=
  IsCyclotomicExtension.norm_zeta_sub_one_of_isPrimePow

@[deprecated (since := "2024-04-02")]
  alias IsCyclotomicExtension.prime_ne_two_pow_norm_zeta_pow_sub_one :=
    IsCyclotomicExtension.norm_zeta_pow_sub_one_of_prime_pow_ne_two

@[deprecated (since := "2024-04-02")]
  alias IsCyclotomicExtension.prime_ne_two_pow_norm_zeta_sub_one :=
    IsCyclotomicExtension.norm_zeta_pow_sub_one_of_prime_ne_two

@[deprecated (since := "2024-04-02")] alias IsCyclotomicExtension.prime_ne_two_norm_zeta_sub_one :=
  IsCyclotomicExtension.norm_zeta_sub_one_of_prime_ne_two

@[deprecated (since := "2024-04-02")] alias IsCyclotomicExtension.two_pow_norm_zeta_sub_one :=
  IsCyclotomicExtension.norm_zeta_pow_sub_one_two


