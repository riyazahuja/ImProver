/-- `IsPrimitiveRoot.autToPow` is injective in the case that it's considered over a cyclotomic
field extension. -/
theorem autToPow_injective : Function.Injective <| hμ.autToPow K := by
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    ⊢ Function.Injective ⇑(IsPrimitiveRoot.autToPow K hμ)
  -/
  intro f g hfg
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hfg : Eq ((IsPrimitiveRoot.autToPow K hμ) f) ((IsPrimitiveRoot.autToPow K hμ) g)
    ⊢ Eq f g
  -/
  apply_fun Units.val at hfg
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hfg : Eq ↑((IsPrimitiveRoot.autToPow K hμ) f) ↑((IsPrimitiveRoot.autToPow K hμ …
    ⊢ Eq f g
  -/
  simp only [IsPrimitiveRoot.coe_autToPow_apply] at hfg
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hfg : Eq ↑⋯.choose ↑⋯.choose
    ⊢ Eq f g
  -/
  generalize_proofs hn₀ hf' hg' at hfg
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : Eq ↑hf'.choose ↑hg'.choose
    ⊢ Eq f g
  -/
  have hf := hf'.choose_spec
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : Eq ↑hf'.choose ↑hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    ⊢ Eq f g
  -/
  have hg := hg'.choose_spec
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : Eq ↑hf'.choose ↑hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq f g
  -/
  generalize_proofs hζ at hf hg
  suffices f (hμ.toRootsOfUnity : Lˣ) = g (hμ.toRootsOfUnity : Lˣ) by
    apply AlgEquiv.coe_algHom_injective
    apply (hμ.powerBasis K).algHom_ext
    exact this
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : Eq ↑hf'.choose ↑hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (f ↑↑hμ.toRootsOfUnity) (g ↑↑hμ.toRootsOfUnity)
  -/
  rw [ZMod.eq_iff_modEq_nat] at hfg
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (f ↑↑hμ.toRootsOfUnity) (g ↑↑hμ.toRootsOfUnity)
  -/
  refine (hf.trans ?_).trans hg.symm
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose) (HPow.hPow (↑↑hμ.toRootsOfUn …
  -/
  rw [← rootsOfUnity.coe_pow _ hf'.choose, ← rootsOfUnity.coe_pow _ hg'.choose]
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq ↑↑(HPow.hPow hμ.toRootsOfUnity hf'.choose) ↑↑(HPow.hPow hμ.toRootsOfUnity …
  -/
  congr 2
  /-
    case e_self.e_self
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (HPow.hPow hμ.toRootsOfUnity hf'.choose) (HPow.hPow hμ.toRootsOfUnity hg' …
  -/
  rw [pow_eq_pow_iff_modEq]
  /-
    case e_self.e_self
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ (orderOf hμ.toRootsOfUnity).ModEq hf'.choose hg'.choose
  -/
  convert hfg
  -- Porting note: was `{occs := occurrences.pos [2]}` (for the second rewrite)
  /-
    case h.e'_1
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (orderOf hμ.toRootsOfUnity) ↑n
  -/
  conv => enter [2]; rw [hμ.eq_orderOf, ← hμ.val_toRootsOfUnity_coe]
  /-
    case h.e'_1
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    f g : AlgEquiv K L L
    hn₀ : NeZero ↑n
    hf' : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hg' : Exists fun m => Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUni …
    hfg : (↑n).ModEq hf'.choose hg'.choose
    hf : Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hf'.choose)
    hg : Eq (g ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity) hg'.choose)
    ⊢ Eq (orderOf hμ.toRootsOfUnity) (orderOf ↑↑hμ.toRootsOfUnity)
  -/
  rw [orderOf_units, Subgroup.orderOf_coe]
  /-
    🎉 no goals
  -/


/-- Cyclotomic extensions are abelian. -/
noncomputable def Aut.commGroup : CommGroup (L ≃ₐ[K] L) :=
  ((zeta_spec n K L).autToPow_injective K).commGroup _ (map_one _) (map_mul _) (map_inv _)
    (map_div _) (map_pow _) (map_zpow _)


/-- The `MulEquiv` that takes an automorphism `f` to the element `k : (ZMod n)ˣ` such that
  `f μ = μ ^ k` for any root of unity `μ`. A strengthening of `IsPrimitiveRoot.autToPow`. -/
@[simps]
noncomputable def autEquivPow (h : Irreducible (cyclotomic n K)) : (L ≃ₐ[K] L) ≃* (ZMod n)ˣ :=
  let hζ := zeta_spec n K L
  let hμ t := hζ.pow_of_coprime _ (ZMod.val_coe_unit_coprime t)
  { (zeta_spec n K L).autToPow K with
    invFun := fun t =>
      (hζ.powerBasis K).equivOfMinpoly ((hμ t).powerBasis K)
        (by
          /-
            n : PNat
            K : Type u_1
            inst✝⁴ : Field K
            L : Type u_2
            μ : L
            inst✝³ : CommRing L
            inst✝² : IsDomain L
            hμ✝ : IsPrimitiveRoot μ ↑n
            inst✝¹ : Algebra K L
            inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
            h : Irreducible (Polynomial.cyclotomic (↑n) K)
            hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
            hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
            t : Units (ZMod ↑n)
            ⊢ Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimitive …
          -/
          haveI := IsCyclotomicExtension.neZero' n K L
          /-
            n : PNat
            K : Type u_1
            inst✝⁴ : Field K
            L : Type u_2
            μ : L
            inst✝³ : CommRing L
            inst✝² : IsDomain L
            hμ✝ : IsPrimitiveRoot μ ↑n
            inst✝¹ : Algebra K L
            inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
            h : Irreducible (Polynomial.cyclotomic (↑n) K)
            hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
            hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
            t : Units (ZMod ↑n)
            this : NeZero ↑↑n
            ⊢ Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimitive …
          -/
          simp only [IsPrimitiveRoot.powerBasis_gen]
          have hr :=
            IsPrimitiveRoot.minpoly_eq_cyclotomic_of_irreducible
              ((zeta_spec n K L).pow_of_coprime _ (ZMod.val_coe_unit_coprime t)) h
          /-
            n : PNat
            K : Type u_1
            inst✝⁴ : Field K
            L : Type u_2
            μ : L
            inst✝³ : CommRing L
            inst✝² : IsDomain L
            hμ✝ : IsPrimitiveRoot μ ↑n
            inst✝¹ : Algebra K L
            inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
            h : Irreducible (Polynomial.cyclotomic (↑n) K)
            hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
            hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
            t : Units (ZMod ↑n)
            this : NeZero ↑↑n
            hr : Eq (Polynomial.cyclotomic (↑n) K) (minpoly K (HPow.hPow (IsCyclotomicExte …
            ⊢ Eq (minpoly K (IsCyclotomicExtension.zeta n K L)) (minpoly K (HPow.hPow (IsC …
          -/
          exact ((zeta_spec n K L).minpoly_eq_cyclotomic_of_irreducible h).symm.trans hr)
          /-
            🎉 no goals
          -/
    left_inv := fun f => by
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq ((fun t => (IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveR …
      -/
      simp only [MonoidHom.toFun_eq_coe]
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq ((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.powerB …
      -/
      apply AlgEquiv.coe_algHom_injective
      /-
        case a
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq ↑((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.power …
      -/
      apply (hζ.powerBasis K).algHom_ext
-- Porting note: the proof is slightly different because of coercions.
      /-
        case a.h
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq (↑((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.powe …
      -/
      simp only [AlgHom.coe_coe]
      /-
        case a.h
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.power …
      -/
      rw [PowerBasis.equivOfMinpoly_gen]
      /-
        case a.h
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        f : AlgEquiv K L L
        ⊢ Eq (IsPrimitiveRoot.powerBasis K ⋯).gen (f (IsPrimitiveRoot.powerBasis K hζ) …
      -/
      simp only [IsPrimitiveRoot.powerBasis_gen, IsPrimitiveRoot.autToPow_spec]
      /-
        🎉 no goals
      -/
    right_inv := fun x => by
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        ⊢ Eq ((↑__src✝).toFun ((fun t => (IsPrimitiveRoot.powerBasis K hζ).equivOfMinp …
      -/
      simp only [MonoidHom.toFun_eq_coe]
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        ⊢ Eq ((IsPrimitiveRoot.autToPow K ⋯) ((IsPrimitiveRoot.powerBasis K hζ).equivO …
      -/
      generalize_proofs _ _ h
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      have key := hζ.autToPow_spec K ((hζ.powerBasis K).equivOfMinpoly ((hμ x).powerBasis K) h)
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        key : Eq (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑((IsPrimitiveRoot.aut …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      have := (hζ.powerBasis K).equivOfMinpoly_gen ((hμ x).powerBasis K) h
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        key : Eq (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑((IsPrimitiveRoot.aut …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      rw [hζ.powerBasis_gen K] at this
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        key : Eq (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑((IsPrimitiveRoot.aut …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      rw [this, IsPrimitiveRoot.powerBasis_gen] at key
-- Porting note: was `rw ← hζ.coe_to_roots_of_unity_coe at key {occs := occurrences.pos [1, 5]}`.
      conv at key =>
        congr; congr
        rw [← hζ.val_toRootsOfUnity_coe]
        rfl; rfl
        rw [← hζ.val_toRootsOfUnity_coe]
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        key : Eq (HPow.hPow (↑↑hζ.toRootsOfUnity) (↑((IsPrimitiveRoot.autToPow K hζ) ( …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      simp only [← rootsOfUnity.coe_pow] at key
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        key : Eq ↑↑(HPow.hPow hζ.toRootsOfUnity (↑((IsPrimitiveRoot.autToPow K hζ) ((I …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      replace key := rootsOfUnity.coe_injective key
      rw [pow_eq_pow_iff_modEq, ← Subgroup.orderOf_coe, ← orderOf_units, hζ.val_toRootsOfUnity_coe,
        ← (zeta_spec n K L).eq_orderOf, ← ZMod.eq_iff_modEq_nat] at key
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        key : Eq ↑(↑((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      simp only [ZMod.natCast_val, ZMod.cast_id', id] at key
      /-
        n : PNat
        K : Type u_1
        inst✝⁴ : Field K
        L : Type u_2
        μ : L
        inst✝³ : CommRing L
        inst✝² : IsDomain L
        hμ✝ : IsPrimitiveRoot μ ↑n
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
        hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n := IsCyclotomicExte …
        hμ : ∀ (t : Units (ZMod ↑n)), IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtensio …
        x : Units (ZMod ↑n)
        pf✝¹ : NeZero ↑n
        pf✝ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑x).val) ↑n
        h : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimiti …
        this : Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot. …
        key : Eq ↑((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ). …
        ⊢ Eq ((IsPrimitiveRoot.autToPow K hζ) ((IsPrimitiveRoot.powerBasis K hζ).equiv …
      -/
      exact Units.ext key }
      /-
        🎉 no goals
      -/


/-- Maps `μ` to the `AlgEquiv` that sends `IsCyclotomicExtension.zeta` to `μ`. -/
noncomputable def fromZetaAut : L ≃ₐ[K] L :=
  let hζ := (zeta_spec n K L).eq_pow_of_pow_eq_one hμ.pow_eq_one
  (autEquivPow L h).symm <|
    ZMod.unitOfCoprime hζ.choose <|
      ((zeta_spec n K L).pow_iff_coprime n.pos hζ.choose).mp <| hζ.choose_spec.2.symm ▸ hμ


theorem fromZetaAut_spec : fromZetaAut hμ h (zeta n K L) = μ := by
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq ((IsCyclotomicExtension.fromZetaAut hμ h) (IsCyclotomicExtension.zeta n K …
  -/
  simp_rw [fromZetaAut, autEquivPow_symm_apply]
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h : Irreducible (Polynomial.cyclotomic (↑n) K)
    ⊢ Eq (((IsPrimitiveRoot.powerBasis K ⋯).equivOfMinpoly (IsPrimitiveRoot.powerB …
  -/
  generalize_proofs hζ h _ hμ _
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ✝ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n
    h : Exists fun i => And (LT.lt i ↑n) (Eq (HPow.hPow (IsCyclotomicExtension.zet …
    pf✝¹ : h.choose.Coprime ↑n
    hμ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑(ZMod.uni …
    pf✝ : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimi …
    ⊢ Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.power …
  -/
  nth_rewrite 4 [← hζ.powerBasis_gen K]
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ✝ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n
    h : Exists fun i => And (LT.lt i ↑n) (Eq (HPow.hPow (IsCyclotomicExtension.zet …
    pf✝¹ : h.choose.Coprime ↑n
    hμ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑(ZMod.uni …
    pf✝ : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimi …
    ⊢ Eq (((IsPrimitiveRoot.powerBasis K hζ).equivOfMinpoly (IsPrimitiveRoot.power …
  -/
  rw [PowerBasis.equivOfMinpoly_gen, hμ.powerBasis_gen K]
  /-
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ✝ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n
    h : Exists fun i => And (LT.lt i ↑n) (Eq (HPow.hPow (IsCyclotomicExtension.zet …
    pf✝¹ : h.choose.Coprime ↑n
    hμ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑(ZMod.uni …
    pf✝ : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimi …
    ⊢ Eq (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑(ZMod.unitOfCoprime h.cho …
  -/
  convert h.choose_spec.2
  /-
    case h.e'_2.h.e'_6
    n : PNat
    K : Type u_1
    inst✝⁴ : Field K
    L : Type u_2
    μ : L
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    hμ✝ : IsPrimitiveRoot μ ↑n
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
    h✝ : Irreducible (Polynomial.cyclotomic (↑n) K)
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta n K L) ↑n
    h : Exists fun i => And (LT.lt i ↑n) (Eq (HPow.hPow (IsCyclotomicExtension.zet …
    pf✝¹ : h.choose.Coprime ↑n
    hμ : IsPrimitiveRoot (HPow.hPow (IsCyclotomicExtension.zeta n K L) (↑(ZMod.uni …
    pf✝ : Eq (minpoly K (IsPrimitiveRoot.powerBasis K hζ).gen) (minpoly K (IsPrimi …
    ⊢ Eq (↑(ZMod.unitOfCoprime h.choose pf✝¹)).val h.choose
  -/
  exact ZMod.val_cast_of_lt h.choose_spec.1
  /-
    🎉 no goals
  -/


/-- `IsCyclotomicExtension.autEquivPow` repackaged in terms of `Gal`.
Asserts that the Galois group of `cyclotomic n K` is equivalent to `(ZMod n)ˣ`
if `cyclotomic n K` is irreducible in the base field. -/
noncomputable def galCyclotomicEquivUnitsZMod : (cyclotomic n K).Gal ≃* (ZMod n)ˣ :=
  (AlgEquiv.autCongr
          (IsSplittingField.algEquiv L _ : L ≃ₐ[K] (cyclotomic n K).SplittingField)).symm.trans
    (IsCyclotomicExtension.autEquivPow L h)


/-- `IsCyclotomicExtension.autEquivPow` repackaged in terms of `Gal`.
Asserts that the Galois group of `X ^ n - 1` is equivalent to `(ZMod n)ˣ`
if `cyclotomic n K` is irreducible in the base field. -/
noncomputable def galXPowEquivUnitsZMod : (X ^ (n : ℕ) - 1 : K[X]).Gal ≃* (ZMod n)ˣ :=
  (AlgEquiv.autCongr
      (IsSplittingField.algEquiv L _ : L ≃ₐ[K] (X ^ (n : ℕ) - 1 : K[X]).SplittingField)).symm.trans
    (IsCyclotomicExtension.autEquivPow L h)


