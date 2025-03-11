/-- The Galois group of a polynomial. -/
def Gal :=
  p.SplittingField ≃ₐ[F] p.SplittingField
-- Porting note(https://github.com/leanprover-community/mathlib4/issues/5020):
-- deriving Group, Fintype


instance instGroup : Group (Gal p) :=
  inferInstanceAs (Group (p.SplittingField ≃ₐ[F] p.SplittingField))


instance instFintype : Fintype (Gal p) :=
  inferInstanceAs (Fintype (p.SplittingField ≃ₐ[F] p.SplittingField))


instance : EquivLike p.Gal p.SplittingField p.SplittingField :=
  inferInstanceAs (EquivLike (p.SplittingField ≃ₐ[F] p.SplittingField) _ _)


instance : AlgEquivClass p.Gal F p.SplittingField p.SplittingField :=
  inferInstanceAs (AlgEquivClass (p.SplittingField ≃ₐ[F] p.SplittingField) F _ _)


instance applyMulSemiringAction : MulSemiringAction p.Gal p.SplittingField :=
  AlgEquiv.applyMulSemiringAction


@[ext]
theorem ext {σ τ : p.Gal} (h : ∀ x ∈ p.rootSet p.SplittingField, σ x = τ x) : σ = τ := by
  refine
    AlgEquiv.ext fun x =>
      (AlgHom.mem_equalizer σ.toAlgHom τ.toAlgHom x).mp
        ((SetLike.ext_iff.mp ?_ x).mpr Algebra.mem_top)
  /-
    F : Type u_1
    inst✝ : Field F
    p : Polynomial F
    σ τ : p.Gal
    h : ∀ (x : p.SplittingField), Membership.mem (p.rootSet p.SplittingField) x →  …
    x : p.SplittingField
    ⊢ Eq (AlgHom.equalizer ↑σ ↑τ) Top.top
  -/
  rwa [eq_top_iff, ← SplittingField.adjoin_rootSet, Algebra.adjoin_le_iff]
  /-
    🎉 no goals
  -/


/-- If `p` splits in `F` then the `p.gal` is trivial. -/
def uniqueGalOfSplits (h : p.Splits (RingHom.id F)) : Unique p.Gal where
  default := 1
  uniq f :=
    AlgEquiv.ext fun x => by
      obtain ⟨y, rfl⟩ :=
        Algebra.mem_bot.mp
          ((SetLike.ext_iff.mp ((IsSplittingField.splits_iff _ p).mp h) x).mp Algebra.mem_top)
      /-
        case intro
        F : Type u_1
        inst✝² : Field F
        p q : Polynomial F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        h : Polynomial.Splits (RingHom.id F) p
        f : p.Gal
        y : F
        ⊢ Eq (f ((algebraMap F p.SplittingField) y)) (Inhabited.default ((algebraMap F …
      -/
      rw [AlgEquiv.commutes, AlgEquiv.commutes]
      /-
        🎉 no goals
      -/


instance [h : Fact (p.Splits (RingHom.id F))] : Unique p.Gal :=
  uniqueGalOfSplits _ h.1


instance uniqueGalZero : Unique (0 : F[X]).Gal :=
  uniqueGalOfSplits _ (splits_zero _)


instance uniqueGalOne : Unique (1 : F[X]).Gal :=
  uniqueGalOfSplits _ (splits_one _)


instance uniqueGalC (x : F) : Unique (C x).Gal :=
  uniqueGalOfSplits _ (splits_C _ _)


instance uniqueGalX : Unique (X : F[X]).Gal :=
  uniqueGalOfSplits _ (splits_X _)


instance uniqueGalXSubC (x : F) : Unique (X - C x).Gal :=
  uniqueGalOfSplits _ (splits_X_sub_C _)


instance uniqueGalXPow (n : ℕ) : Unique (X ^ n : F[X]).Gal :=
  uniqueGalOfSplits _ (splits_X_pow _ _)


instance [h : Fact (p.Splits (algebraMap F E))] : Algebra p.SplittingField E :=
  (IsSplittingField.lift p.SplittingField p h.1).toRingHom.toAlgebra


instance [h : Fact (p.Splits (algebraMap F E))] : IsScalarTower F p.SplittingField E :=
  IsScalarTower.of_algebraMap_eq fun x =>
    ((IsSplittingField.lift p.SplittingField p h.1).commutes x).symm

-- The `Algebra p.SplittingField E` instance above behaves badly when
-- `E := p.SplittingField`, since it may result in a unification problem
-- `IsSplittingField.lift.toRingHom.toAlgebra =?= Algebra.id`,
-- which takes an extremely long time to resolve, causing timeouts.
-- Since we don't really care about this definition, marking it as irreducible
-- causes that unification to error out early.

/-- Restrict from a superfield automorphism into a member of `gal p`. -/
def restrict [Fact (p.Splits (algebraMap F E))] : (E ≃ₐ[F] E) →* p.Gal :=
  AlgEquiv.restrictNormalHom p.SplittingField


theorem restrict_surjective [Fact (p.Splits (algebraMap F E))] [Normal F E] :
    Function.Surjective (restrict p E) :=
  AlgEquiv.restrictNormalHom_surjective E


/-- The function taking `rootSet p p.SplittingField` to `rootSet p E`. This is actually a bijection,
see `Polynomial.Gal.mapRoots_bijective`. -/
def mapRoots [Fact (p.Splits (algebraMap F E))] : rootSet p p.SplittingField → rootSet p E :=
  Set.MapsTo.restrict (IsScalarTower.toAlgHom F p.SplittingField E) _ _ <| rootSet_mapsTo _


theorem mapRoots_bijective [h : Fact (p.Splits (algebraMap F E))] :
    Function.Bijective (mapRoots p E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    p : Polynomial F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : Fact (Polynomial.Splits (algebraMap F E) p)
    ⊢ Function.Bijective (Polynomial.Gal.mapRoots p E)
  -/
  constructor
    /-
      case left
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      ⊢ Function.Injective (Polynomial.Gal.mapRoots p E)
    -/
  · exact fun _ _ h => Subtype.ext (RingHom.injective _ (Subtype.ext_iff.mp h))
    /-
      🎉 no goals
    -/
    /-
      case right
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      ⊢ Function.Surjective (Polynomial.Gal.mapRoots p E)
    -/
  · intro y
    -- this is just an equality of two different ways to write the roots of `p` as an `E`-polynomial
    have key :=
      roots_map (IsScalarTower.toAlgHom F p.SplittingField E : p.SplittingField →+* E)
        ((splits_id_iff_splits _).mpr (IsSplittingField.splits p.SplittingField p))
    /-
      case right
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      y : ↑(p.rootSet E)
      key : Eq (Polynomial.map (↑(IsScalarTower.toAlgHom F p.SplittingField E)) (Pol …
      ⊢ Exists fun a => Eq (Polynomial.Gal.mapRoots p E a) y
    -/
    rw [map_map, AlgHom.comp_algebraMap] at key
    /-
      case right
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      y : ↑(p.rootSet E)
      key : Eq (Polynomial.map (algebraMap F E) p).roots (Multiset.map (⇑↑(IsScalarT …
      ⊢ Exists fun a => Eq (Polynomial.Gal.mapRoots p E a) y
    -/
    have hy := Subtype.mem y
    /-
      case right
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      y : ↑(p.rootSet E)
      key : Eq (Polynomial.map (algebraMap F E) p).roots (Multiset.map (⇑↑(IsScalarT …
      hy : Membership.mem (p.rootSet E) ↑y
      ⊢ Exists fun a => Eq (Polynomial.Gal.mapRoots p E a) y
    -/
    simp only [rootSet, Finset.mem_coe, Multiset.mem_toFinset, key, Multiset.mem_map] at hy
    /-
      case right
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      y : ↑(p.rootSet E)
      key : Eq (Polynomial.map (algebraMap F E) p).roots (Multiset.map (⇑↑(IsScalarT …
      hy : Exists fun a => And (Membership.mem (Polynomial.map (algebraMap F p.Split …
      ⊢ Exists fun a => Eq (Polynomial.Gal.mapRoots p E a) y
    -/
    rcases hy with ⟨x, hx1, hx2⟩
    /-
      case right.intro.intro
      F : Type u_1
      inst✝² : Field F
      p : Polynomial F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : Fact (Polynomial.Splits (algebraMap F E) p)
      y : ↑(p.rootSet E)
      key : Eq (Polynomial.map (algebraMap F E) p).roots (Multiset.map (⇑↑(IsScalarT …
      x : p.SplittingField
      hx1 : Membership.mem (Polynomial.map (algebraMap F p.SplittingField) p).roots x
      hx2 : Eq (↑(IsScalarTower.toAlgHom F p.SplittingField E) x) ↑y
      ⊢ Exists fun a => Eq (Polynomial.Gal.mapRoots p E a) y
    -/
    exact ⟨⟨x, (@Multiset.mem_toFinset _ (Classical.decEq _) _ _).mpr hx1⟩, Subtype.ext hx2⟩
    /-
      🎉 no goals
    -/


/-- The bijection between `rootSet p p.SplittingField` and `rootSet p E`. -/
def rootsEquivRoots [Fact (p.Splits (algebraMap F E))] : rootSet p p.SplittingField ≃ rootSet p E :=
  Equiv.ofBijective (mapRoots p E) (mapRoots_bijective p E)


instance galActionAux : MulAction p.Gal (rootSet p p.SplittingField) where
  smul ϕ := Set.MapsTo.restrict ϕ _ _ <| rootSet_mapsTo ϕ.toAlgHom
                   /-
                     F : Type u_1
                     inst✝² : Field F
                     p q : Polynomial F
                     E : Type u_2
                     inst✝¹ : Field E
                     inst✝ : Algebra F E
                     x✝ : ↑(p.rootSet p.SplittingField)
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                       /-
                         F : Type u_1
                         inst✝² : Field F
                         p q : Polynomial F
                         E : Type u_2
                         inst✝¹ : Field E
                         inst✝ : Algebra F E
                         x✝² x✝¹ : p.Gal
                         x✝ : ↑(p.rootSet p.SplittingField)
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by ext; rfl
                            /-
                              🎉 no goals
                            -/

-- Porting note: split out from `galAction` below to allow using `smul_def` there.

instance smul [Fact (p.Splits (algebraMap F E))] : SMul p.Gal (rootSet p E) where
  smul ϕ x := rootsEquivRoots p E (ϕ • (rootsEquivRoots p E).symm x)


theorem smul_def [Fact (p.Splits (algebraMap F E))] (ϕ : p.Gal) (x : rootSet p E) :
    ϕ • x = rootsEquivRoots p E (ϕ • (rootsEquivRoots p E).symm x) :=
  rfl


/-- The action of `gal p` on the roots of `p` in `E`. -/
instance galAction [Fact (p.Splits (algebraMap F E))] : MulAction p.Gal (rootSet p E) where
                   /-
                     F : Type u_1
                     inst✝³ : Field F
                     p q : Polynomial F
                     E : Type u_2
                     inst✝² : Field E
                     inst✝¹ : Algebra F E
                     inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
                     x✝ : ↑(p.rootSet E)
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by simp only [smul_def, Equiv.apply_symm_apply, one_smul]
                   /-
                     🎉 no goals
                   -/
  mul_smul _ _ _ := by
    /-
      F : Type u_1
      inst✝³ : Field F
      p q : Polynomial F
      E : Type u_2
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
      x✝² x✝¹ : p.Gal
      x✝ : ↑(p.rootSet E)
      ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
    -/
    simp only [smul_def, Equiv.apply_symm_apply, Equiv.symm_apply_apply, mul_smul]
    /-
      🎉 no goals
    -/


lemma galAction_isPretransitive [Fact (p.Splits (algebraMap F E))] (hp : Irreducible p) :
    MulAction.IsPretransitive p.Gal (p.rootSet E) := by
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    hp : Irreducible p
    ⊢ MulAction.IsPretransitive p.Gal ↑(p.rootSet E)
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    hp : Irreducible p
    x y : ↑(p.rootSet E)
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  have hx := minpoly.eq_of_irreducible hp (mem_rootSet.mp ((rootsEquivRoots p E).symm x).2).2
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    hp : Irreducible p
    x y : ↑(p.rootSet E)
    hx : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly F ↑((Po …
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  have hy := minpoly.eq_of_irreducible hp (mem_rootSet.mp ((rootsEquivRoots p E).symm y).2).2
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    hp : Irreducible p
    x y : ↑(p.rootSet E)
    hx : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly F ↑((Po …
    hy : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly F ↑((Po …
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨g, hg⟩ := (Normal.minpoly_eq_iff_mem_orbit p.SplittingField).mp (hy.symm.trans hx)
  /-
    case intro
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    hp : Irreducible p
    x y : ↑(p.rootSet E)
    hx : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly F ↑((Po …
    hy : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly F ↑((Po …
    g : AlgEquiv F p.SplittingField p.SplittingField
    hg : Eq ((fun m => HSMul.hSMul m ↑((Polynomial.Gal.rootsEquivRoots p E).symm x …
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  exact ⟨g, (rootsEquivRoots p E).apply_eq_iff_eq_symm_apply.mpr (Subtype.ext hg)⟩
  /-
    🎉 no goals
  -/


/-- `Polynomial.Gal.restrict p E` is compatible with `Polynomial.Gal.galAction p E`. -/
@[simp]
theorem restrict_smul [Fact (p.Splits (algebraMap F E))] (ϕ : E ≃ₐ[F] E) (x : rootSet p E) :
    ↑(restrict p E ϕ • x) = ϕ x := by
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : AlgEquiv F E E
    x : ↑(p.rootSet E)
    ⊢ Eq (↑(HSMul.hSMul ((Polynomial.Gal.restrict p E) ϕ) x)) (ϕ ↑x)
  -/
  let ψ := AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F p.SplittingField E)
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : AlgEquiv F E E
    x : ↑(p.rootSet E)
    ψ : AlgEquiv F p.SplittingField (Subtype fun x => Membership.mem (IsScalarTowe …
    ⊢ Eq (↑(HSMul.hSMul ((Polynomial.Gal.restrict p E) ϕ) x)) (ϕ ↑x)
  -/
  change ↑(ψ (ψ.symm _)) = ϕ x
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : AlgEquiv F E E
    x : ↑(p.rootSet E)
    ψ : AlgEquiv F p.SplittingField (Subtype fun x => Membership.mem (IsScalarTowe …
    ⊢ Eq (↑(ψ (ψ.symm (↑((↑ϕ).restrictNormalAux p.SplittingField) (↑↑(AlgEquiv.ofI …
  -/
  rw [AlgEquiv.apply_symm_apply ψ]
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : AlgEquiv F E E
    x : ↑(p.rootSet E)
    ψ : AlgEquiv F p.SplittingField (Subtype fun x => Membership.mem (IsScalarTowe …
    ⊢ Eq (↑(↑((↑ϕ).restrictNormalAux p.SplittingField) (↑↑(AlgEquiv.ofInjectiveFie …
  -/
  change ϕ (rootsEquivRoots p E ((rootsEquivRoots p E).symm x)) = ϕ x
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : AlgEquiv F E E
    x : ↑(p.rootSet E)
    ψ : AlgEquiv F p.SplittingField (Subtype fun x => Membership.mem (IsScalarTowe …
    ⊢ Eq (ϕ ↑((Polynomial.Gal.rootsEquivRoots p E) ((Polynomial.Gal.rootsEquivRoot …
  -/
  rw [Equiv.apply_symm_apply (rootsEquivRoots p E)]
  /-
    🎉 no goals
  -/


/-- `Polynomial.Gal.galAction` as a permutation representation -/
def galActionHom [Fact (p.Splits (algebraMap F E))] : p.Gal →* Equiv.Perm (rootSet p E) :=
  MulAction.toPermHom _ _


theorem galActionHom_restrict [Fact (p.Splits (algebraMap F E))] (ϕ : E ≃ₐ[F] E) (x : rootSet p E) :
    ↑(galActionHom p E (restrict p E ϕ) x) = ϕ x :=
  restrict_smul ϕ x


/-- `gal p` embeds as a subgroup of permutations of the roots of `p` in `E`. -/
theorem galActionHom_injective [Fact (p.Splits (algebraMap F E))] :
    Function.Injective (galActionHom p E) := by
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ⊢ Function.Injective ⇑(Polynomial.Gal.galActionHom p E)
  -/
  rw [injective_iff_map_eq_one]
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ⊢ ∀ (a : p.Gal), Eq ((Polynomial.Gal.galActionHom p E) a) 1 → Eq a 1
  -/
  intro ϕ hϕ
  /-
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : p.Gal
    hϕ : Eq ((Polynomial.Gal.galActionHom p E) ϕ) 1
    ⊢ Eq ϕ 1
  -/
  ext (x hx)
  /-
    case h
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : p.Gal
    hϕ : Eq ((Polynomial.Gal.galActionHom p E) ϕ) 1
    x : p.SplittingField
    hx : Membership.mem (p.rootSet p.SplittingField) x
    ⊢ Eq (ϕ x) (1 x)
  -/
  have key := Equiv.Perm.ext_iff.mp hϕ (rootsEquivRoots p E ⟨x, hx⟩)
  change
    rootsEquivRoots p E (ϕ • (rootsEquivRoots p E).symm (rootsEquivRoots p E ⟨x, hx⟩)) =
      rootsEquivRoots p E ⟨x, hx⟩
    at key
  /-
    case h
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : p.Gal
    hϕ : Eq ((Polynomial.Gal.galActionHom p E) ϕ) 1
    x : p.SplittingField
    hx : Membership.mem (p.rootSet p.SplittingField) x
    key : Eq ((Polynomial.Gal.rootsEquivRoots p E) (HSMul.hSMul ϕ ((Polynomial.Gal …
    ⊢ Eq (ϕ x) (1 x)
  -/
  rw [Equiv.symm_apply_apply] at key
  /-
    case h
    F : Type u_1
    inst✝³ : Field F
    p : Polynomial F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Fact (Polynomial.Splits (algebraMap F E) p)
    ϕ : p.Gal
    hϕ : Eq ((Polynomial.Gal.galActionHom p E) ϕ) 1
    x : p.SplittingField
    hx : Membership.mem (p.rootSet p.SplittingField) x
    key : Eq ((Polynomial.Gal.rootsEquivRoots p E) (HSMul.hSMul ϕ ⟨x, hx⟩)) ((Poly …
    ⊢ Eq (ϕ x) (1 x)
  -/
  exact Subtype.ext_iff.mp (Equiv.injective (rootsEquivRoots p E) key)
  /-
    🎉 no goals
  -/


/-- `Polynomial.Gal.restrict`, when both fields are splitting fields of polynomials. -/
def restrictDvd (hpq : p ∣ q) : q.Gal →* p.Gal :=
  haveI := Classical.dec (q = 0)
  if hq : q = 0 then 1
  else
    @restrict F _ p _ _ _
      ⟨splits_of_splits_of_dvd (algebraMap F q.SplittingField) hq (SplittingField.splits q) hpq⟩


theorem restrictDvd_def [Decidable (q = 0)] (hpq : p ∣ q) :
    restrictDvd hpq =
      if hq : q = 0 then 1
      else
        @restrict F _ p _ _ _
          ⟨splits_of_splits_of_dvd (algebraMap F q.SplittingField) hq (SplittingField.splits q)
              hpq⟩ := by
  -- Porting note: added `unfold`
  /-
    F : Type u_1
    inst✝¹ : Field F
    p q : Polynomial F
    inst✝ : Decidable (Eq q 0)
    hpq : Dvd.dvd p q
    ⊢ Eq (Polynomial.Gal.restrictDvd hpq) (dite (Eq q 0) (fun hq => 1) fun hq => P …
  -/
  unfold restrictDvd
  /-
    F : Type u_1
    inst✝¹ : Field F
    p q : Polynomial F
    inst✝ : Decidable (Eq q 0)
    hpq : Dvd.dvd p q
    ⊢ Eq (dite (Eq q 0) (fun hq => 1) fun hq => Polynomial.Gal.restrict p q.Splitt …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


theorem restrictDvd_surjective (hpq : p ∣ q) (hq : q ≠ 0) :
    Function.Surjective (restrictDvd hpq) := by
  classical
    -- Porting note: was `simp only [restrictDvd_def, dif_neg hq, restrict_surjective]`
    haveI := Fact.mk <|
      splits_of_splits_of_dvd (algebraMap F q.SplittingField) hq (SplittingField.splits q) hpq
    simp only [restrictDvd_def, dif_neg hq]
    exact restrict_surjective _ _


/-- The Galois group of a product maps into the product of the Galois groups. -/
def restrictProd : (p * q).Gal →* p.Gal × q.Gal :=
  MonoidHom.prod (restrictDvd (dvd_mul_right p q)) (restrictDvd (dvd_mul_left q p))


/-- `Polynomial.Gal.restrictProd` is actually a subgroup embedding. -/
theorem restrictProd_injective : Function.Injective (restrictProd p q) := by
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    ⊢ Function.Injective ⇑(Polynomial.Gal.restrictProd p q)
  -/
  by_cases hpq : p * q = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      p q : Polynomial F
      hpq : Eq (HMul.hMul p q) 0
      ⊢ Function.Injective ⇑(Polynomial.Gal.restrictProd p q)
    -/
  · have : Unique (p * q).Gal := by rw [hpq]; infer_instance
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      p q : Polynomial F
      hpq : Eq (HMul.hMul p q) 0
      this : Unique (HMul.hMul p q).Gal
      ⊢ Function.Injective ⇑(Polynomial.Gal.restrictProd p q)
    -/
    exact fun f g _ => Eq.trans (Unique.eq_default f) (Unique.eq_default g).symm
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hpq : Not (Eq (HMul.hMul p q) 0)
    ⊢ Function.Injective ⇑(Polynomial.Gal.restrictProd p q)
  -/
  intro f g hfg
  classical
  simp only [restrictProd, restrictDvd_def] at hfg
  simp only [dif_neg hpq, MonoidHom.prod_apply, Prod.mk.inj_iff] at hfg
  ext (x hx)
  rw [rootSet_def, aroots_mul hpq] at hx
  cases' Multiset.mem_add.mp (Multiset.mem_toFinset.mp hx) with h h
  · haveI : Fact (p.Splits (algebraMap F (p * q).SplittingField)) :=
      ⟨splits_of_splits_of_dvd _ hpq (SplittingField.splits (p * q)) (dvd_mul_right p q)⟩
    have key :
      x =
        algebraMap p.SplittingField (p * q).SplittingField
          ((rootsEquivRoots p _).invFun
            ⟨x, (@Multiset.mem_toFinset _ (Classical.decEq _) _ _).mpr h⟩) :=
      Subtype.ext_iff.mp (Equiv.apply_symm_apply (rootsEquivRoots p _) ⟨x, _⟩).symm
    rw [key, ← AlgEquiv.restrictNormal_commutes, ← AlgEquiv.restrictNormal_commutes]
    exact congr_arg _ (AlgEquiv.ext_iff.mp hfg.1 _)
  · haveI : Fact (q.Splits (algebraMap F (p * q).SplittingField)) :=
      ⟨splits_of_splits_of_dvd _ hpq (SplittingField.splits (p * q)) (dvd_mul_left q p)⟩
    have key :
      x =
        algebraMap q.SplittingField (p * q).SplittingField
          ((rootsEquivRoots q _).invFun
            ⟨x, (@Multiset.mem_toFinset _ (Classical.decEq _) _ _).mpr h⟩) :=
      Subtype.ext_iff.mp (Equiv.apply_symm_apply (rootsEquivRoots q _) ⟨x, _⟩).symm
    rw [key, ← AlgEquiv.restrictNormal_commutes, ← AlgEquiv.restrictNormal_commutes]
    exact congr_arg _ (AlgEquiv.ext_iff.mp hfg.2 _)


theorem mul_splits_in_splittingField_of_mul {p₁ q₁ p₂ q₂ : F[X]} (hq₁ : q₁ ≠ 0) (hq₂ : q₂ ≠ 0)
    (h₁ : p₁.Splits (algebraMap F q₁.SplittingField))
    (h₂ : p₂.Splits (algebraMap F q₂.SplittingField)) :
    (p₁ * p₂).Splits (algebraMap F (q₁ * q₂).SplittingField) := by
  /-
    F : Type u_1
    inst✝ : Field F
    p₁ q₁ p₂ q₂ : Polynomial F
    hq₁ : Ne q₁ 0
    hq₂ : Ne q₂ 0
    h₁ : Polynomial.Splits (algebraMap F q₁.SplittingField) p₁
    h₂ : Polynomial.Splits (algebraMap F q₂.SplittingField) p₂
    ⊢ Polynomial.Splits (algebraMap F (HMul.hMul q₁ q₂).SplittingField) (HMul.hMul …
  -/
  apply splits_mul
  · rw [←
      (SplittingField.lift q₁
          (splits_of_splits_of_dvd (algebraMap F (q₁ * q₂).SplittingField) (mul_ne_zero hq₁ hq₂)
            (SplittingField.splits _) (dvd_mul_right q₁ q₂))).comp_algebraMap]
    /-
      case hf
      F : Type u_1
      inst✝ : Field F
      p₁ q₁ p₂ q₂ : Polynomial F
      hq₁ : Ne q₁ 0
      hq₂ : Ne q₂ 0
      h₁ : Polynomial.Splits (algebraMap F q₁.SplittingField) p₁
      h₂ : Polynomial.Splits (algebraMap F q₂.SplittingField) p₂
      ⊢ Polynomial.Splits ((↑(Polynomial.SplittingField.lift q₁ ⋯)).comp (algebraMap …
    -/
    exact splits_comp_of_splits _ _ h₁
    /-
      🎉 no goals
    -/
  · rw [←
      (SplittingField.lift q₂
          (splits_of_splits_of_dvd (algebraMap F (q₁ * q₂).SplittingField) (mul_ne_zero hq₁ hq₂)
            (SplittingField.splits _) (dvd_mul_left q₂ q₁))).comp_algebraMap]
    /-
      case hg
      F : Type u_1
      inst✝ : Field F
      p₁ q₁ p₂ q₂ : Polynomial F
      hq₁ : Ne q₁ 0
      hq₂ : Ne q₂ 0
      h₁ : Polynomial.Splits (algebraMap F q₁.SplittingField) p₁
      h₂ : Polynomial.Splits (algebraMap F q₂.SplittingField) p₂
      ⊢ Polynomial.Splits ((↑(Polynomial.SplittingField.lift q₂ ⋯)).comp (algebraMap …
    -/
    exact splits_comp_of_splits _ _ h₂
    /-
      🎉 no goals
    -/


/-- `p` splits in the splitting field of `p ∘ q`, for `q` non-constant. -/
theorem splits_in_splittingField_of_comp (hq : q.natDegree ≠ 0) :
    p.Splits (algebraMap F (p.comp q).SplittingField) := by
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hq : Ne q.natDegree 0
    ⊢ Polynomial.Splits (algebraMap F (p.comp q).SplittingField) p
  -/
  let P : F[X] → Prop := fun r => r.Splits (algebraMap F (r.comp q).SplittingField)
  have key1 : ∀ {r : F[X]}, Irreducible r → P r := by
    intro r hr
    by_cases hr' : natDegree r = 0
    · exact splits_of_natDegree_le_one _ (le_trans (le_of_eq hr') zero_le_one)
    obtain ⟨x, hx⟩ :=
      exists_root_of_splits _ (SplittingField.splits (r.comp q)) fun h =>
        hr'
          ((mul_eq_zero.mp
                (natDegree_comp.symm.trans (natDegree_eq_of_degree_eq_some h))).resolve_right
            hq)
    rw [← aeval_def, aeval_comp] at hx
    have h_normal : Normal F (r.comp q).SplittingField := SplittingField.instNormal (r.comp q)
    have qx_int := Normal.isIntegral h_normal (aeval x q)
    exact
      splits_of_splits_of_dvd _ (minpoly.ne_zero qx_int) (Normal.splits h_normal _)
        ((minpoly.irreducible qx_int).dvd_symm hr (minpoly.dvd F _ hx))
  have key2 : ∀ {p₁ p₂ : F[X]}, P p₁ → P p₂ → P (p₁ * p₂) := by
    intro p₁ p₂ hp₁ hp₂
    by_cases h₁ : p₁.comp q = 0
    · cases' comp_eq_zero_iff.mp h₁ with h h
      · rw [h, zero_mul]
        exact splits_zero _
      · exact False.elim (hq (by rw [h.2, natDegree_C]))
    by_cases h₂ : p₂.comp q = 0
    · cases' comp_eq_zero_iff.mp h₂ with h h
      · rw [h, mul_zero]
        exact splits_zero _
      · exact False.elim (hq (by rw [h.2, natDegree_C]))
    have key := mul_splits_in_splittingField_of_mul h₁ h₂ hp₁ hp₂
    rwa [← mul_comp] at key
  -- Porting note: the last part of the proof needs to be unfolded to avoid timeout
  -- original proof
  -- exact
  --  WfDvdMonoid.induction_on_irreducible p (splits_zero _) (fun _ => splits_of_isUnit _)
  --    fun _ _ _ h => key2 (key1 h)
  induction p using WfDvdMonoid.induction_on_irreducible with
  | h0 => exact splits_zero _
  | hu u hu => exact splits_of_isUnit (algebraMap F (SplittingField (comp u q))) hu
  -- Porting note: using `exact` instead of `apply` times out
  | hi p₁ p₂ _ hp₂ hp₁ => apply key2 (key1 hp₂) hp₁


/-- `Polynomial.Gal.restrict` for the composition of polynomials. -/
def restrictComp (hq : q.natDegree ≠ 0) : (p.comp q).Gal →* p.Gal :=
  let h : Fact (Splits (algebraMap F (p.comp q).SplittingField) p) :=
    ⟨splits_in_splittingField_of_comp p q hq⟩
  @restrict F _ p _ _ _ h


theorem restrictComp_surjective (hq : q.natDegree ≠ 0) :
    Function.Surjective (restrictComp p q hq) := by
  -- Porting note: was
  -- simp only [restrictComp, restrict_surjective]
  haveI : Fact (Splits (algebraMap F (SplittingField (comp p q))) p) :=
    ⟨splits_in_splittingField_of_comp p q hq⟩
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hq : Ne q.natDegree 0
    this : Fact (Polynomial.Splits (algebraMap F (p.comp q).SplittingField) p)
    ⊢ Function.Surjective ⇑(Polynomial.Gal.restrictComp p q hq)
  -/
  rw [restrictComp]
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    hq : Ne q.natDegree 0
    this : Fact (Polynomial.Splits (algebraMap F (p.comp q).SplittingField) p)
    ⊢ Function.Surjective ⇑(Polynomial.Gal.restrict p (p.comp q).SplittingField)
  -/
  exact restrict_surjective _ _
  /-
    🎉 no goals
  -/


/-- For a separable polynomial, its Galois group has cardinality
equal to the dimension of its splitting field over `F`. -/
theorem card_of_separable (hp : p.Separable) : Fintype.card p.Gal = finrank F p.SplittingField :=
  haveI : IsGalois F p.SplittingField := IsGalois.of_separable_splitting_field hp
  IsGalois.card_aut_eq_finrank F p.SplittingField


theorem prime_degree_dvd_card [CharZero F] (p_irr : Irreducible p) (p_deg : p.natDegree.Prime) :
    p.natDegree ∣ Fintype.card p.Gal := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    p : Polynomial F
    inst✝ : CharZero F
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    ⊢ Dvd.dvd p.natDegree (Fintype.card p.Gal)
  -/
  rw [Gal.card_of_separable p_irr.separable]
  have hp : p.degree ≠ 0 := fun h =>
    Nat.Prime.ne_zero p_deg (natDegree_eq_zero_iff_degree_le_zero.mpr (le_of_eq h))
  let α : p.SplittingField :=
    rootOfSplits (algebraMap F p.SplittingField) (SplittingField.splits p) hp
  /-
    F : Type u_1
    inst✝¹ : Field F
    p : Polynomial F
    inst✝ : CharZero F
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    hp : Ne p.degree 0
    α : p.SplittingField := Polynomial.rootOfSplits (algebraMap F p.SplittingField …
    ⊢ Dvd.dvd p.natDegree (Module.finrank F p.SplittingField)
  -/
  have hα : IsIntegral F α := .of_finite F α
  /-
    F : Type u_1
    inst✝¹ : Field F
    p : Polynomial F
    inst✝ : CharZero F
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    hp : Ne p.degree 0
    α : p.SplittingField := Polynomial.rootOfSplits (algebraMap F p.SplittingField …
    hα : IsIntegral F α
    ⊢ Dvd.dvd p.natDegree (Module.finrank F p.SplittingField)
  -/
  use Module.finrank F⟮α⟯ p.SplittingField
  suffices (minpoly F α).natDegree = p.natDegree by
    letI _ : AddCommGroup F⟮α⟯ := Ring.toAddCommGroup
    rw [← Module.finrank_mul_finrank F F⟮α⟯ p.SplittingField,
      IntermediateField.adjoin.finrank hα, this]
  suffices minpoly F α ∣ p by
    have key := (minpoly.irreducible hα).dvd_symm p_irr this
    apply le_antisymm
    · exact natDegree_le_of_dvd this p_irr.ne_zero
    · exact natDegree_le_of_dvd key (minpoly.ne_zero hα)
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    p : Polynomial F
    inst✝ : CharZero F
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    hp : Ne p.degree 0
    α : p.SplittingField := Polynomial.rootOfSplits (algebraMap F p.SplittingField …
    hα : IsIntegral F α
    ⊢ Dvd.dvd (minpoly F α) p
  -/
  apply minpoly.dvd F α
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    p : Polynomial F
    inst✝ : CharZero F
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    hp : Ne p.degree 0
    α : p.SplittingField := Polynomial.rootOfSplits (algebraMap F p.SplittingField …
    hα : IsIntegral F α
    ⊢ Eq ((Polynomial.aeval α) p) 0
  -/
  rw [aeval_def, map_rootOfSplits _ (SplittingField.splits p) hp]
  /-
    🎉 no goals
  -/


