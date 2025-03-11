/-- Typeclass characterising splitting fields. -/
@[stacks 09HV "Predicate version"]
class IsSplittingField (f : K[X]) : Prop where
  splits' : Splits (algebraMap K L) f
  adjoin_rootSet' : Algebra.adjoin K (f.rootSet L : Set L) = ⊤


theorem splits (f : K[X]) [IsSplittingField K L f] : Splits (algebraMap K L) f :=
  splits'

-- Porting note: infer kinds are unsupported
-- so we provide a version of `adjoin_rootSet'` with `f` explicit.

theorem adjoin_rootSet (f : K[X]) [IsSplittingField K L f] :
    Algebra.adjoin K (f.rootSet L : Set L) = ⊤ :=
  adjoin_rootSet'


instance map (f : F[X]) [IsSplittingField F L f] : IsSplittingField K L (f.map <| algebraMap F K) :=
      /-
        F : Type u
        K : Type v
        L : Type w
        inst✝⁷ : Field K
        inst✝⁶ : Field L
        inst✝⁵ : Field F
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra F K
        inst✝² : Algebra F L
        inst✝¹ : IsScalarTower F K L
        f : Polynomial F
        inst✝ : Polynomial.IsSplittingField F L f
        ⊢ Polynomial.Splits (algebraMap K L) (Polynomial.map (algebraMap F K) f)
      -/
  ⟨by rw [splits_map_iff, ← IsScalarTower.algebraMap_eq]; exact splits L f,
                                                          /-
                                                            🎉 no goals
                                                          -/
    Subalgebra.restrictScalars_injective F <| by
      rw [rootSet, aroots, map_map, ← IsScalarTower.algebraMap_eq, Subalgebra.restrictScalars_top,
        eq_top_iff, ← adjoin_rootSet L f, Algebra.adjoin_le_iff]
      /-
        F : Type u
        K : Type v
        L : Type w
        inst✝⁷ : Field K
        inst✝⁶ : Field L
        inst✝⁵ : Field F
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra F K
        inst✝² : Algebra F L
        inst✝¹ : IsScalarTower F K L
        f : Polynomial F
        inst✝ : Polynomial.IsSplittingField F L f
        ⊢ HasSubset.Subset (f.rootSet L) ↑(Subalgebra.restrictScalars F (Algebra.adjoi …
      -/
      exact fun x hx => @Algebra.subset_adjoin K _ _ _ _ _ _ hx⟩
      /-
        🎉 no goals
      -/


theorem splits_iff (f : K[X]) [IsSplittingField K L f] :
    Splits (RingHom.id K) f ↔ (⊤ : Subalgebra K L) = ⊥ :=
  ⟨fun h => by -- Porting note: replaced term-mode proof
    rw [eq_bot_iff, ← adjoin_rootSet L f, rootSet, aroots, roots_map (algebraMap K L) h,
      Algebra.adjoin_le_iff]
    /-
      K : Type v
      L : Type w
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      f : Polynomial K
      inst✝ : Polynomial.IsSplittingField K L f
      h : Polynomial.Splits (RingHom.id K) f
      ⊢ HasSubset.Subset ↑(Multiset.map (⇑(algebraMap K L)) f.roots).toFinset ↑Bot.bot
    -/
    intro y hy
    classical
    rw [Multiset.toFinset_map, Finset.mem_coe, Finset.mem_image] at hy
    obtain ⟨x : K, -, hxy : algebraMap K L x = y⟩ := hy
    rw [← hxy]
    exact SetLike.mem_coe.2 <| Subalgebra.algebraMap_mem _ _,
    fun h => @RingEquiv.toRingHom_refl K _ ▸ RingEquiv.self_trans_symm
      (RingEquiv.ofBijective _ <| Algebra.bijective_algebraMap_iff.2 h) ▸ by
        /-
          K : Type v
          L : Type w
          inst✝³ : Field K
          inst✝² : Field L
          inst✝¹ : Algebra K L
          f : Polynomial K
          inst✝ : Polynomial.IsSplittingField K L f
          h : Eq Top.top Bot.bot
          ⊢ Polynomial.Splits ((RingEquiv.ofBijective (algebraMap K L) ⋯).trans (RingEqu …
        -/
        rw [RingEquiv.toRingHom_trans]
        /-
          K : Type v
          L : Type w
          inst✝³ : Field K
          inst✝² : Field L
          inst✝¹ : Algebra K L
          f : Polynomial K
          inst✝ : Polynomial.IsSplittingField K L f
          h : Eq Top.top Bot.bot
          ⊢ Polynomial.Splits ((RingEquiv.ofBijective (algebraMap K L) ⋯).symm.toRingHom …
        -/
        exact splits_comp_of_splits _ _ (splits L f)⟩
        /-
          🎉 no goals
        -/


theorem mul (f g : F[X]) (hf : f ≠ 0) (hg : g ≠ 0) [IsSplittingField F K f]
    [IsSplittingField K L (g.map <| algebraMap F K)] : IsSplittingField F L (f * g) :=
  ⟨(IsScalarTower.algebraMap_eq F K L).symm ▸
      splits_mul _ (splits_comp_of_splits _ _ (splits K f))
        ((splits_map_iff _ _).1 (splits L <| g.map <| algebraMap F K)), by
    classical
    rw [rootSet, aroots_mul (mul_ne_zero hf hg),
      Multiset.toFinset_add, Finset.coe_union, Algebra.adjoin_union_eq_adjoin_adjoin,
      aroots_def, aroots_def, IsScalarTower.algebraMap_eq F K L, ← map_map,
      roots_map (algebraMap K L) ((splits_id_iff_splits <| algebraMap F K).2 <| splits K f),
      Multiset.toFinset_map, Finset.coe_image, Algebra.adjoin_algebraMap, ← rootSet, adjoin_rootSet,
      Algebra.map_top, IsScalarTower.adjoin_range_toAlgHom, ← map_map, ← rootSet, adjoin_rootSet,
      Subalgebra.restrictScalars_top]⟩


open Classical in
/-- Splitting field of `f` embeds into any field that splits `f`. -/
def lift [Algebra K F] (f : K[X]) [IsSplittingField K L f]
    (hf : Splits (algebraMap K F) f) : L →ₐ[K] F :=
  if hf0 : f = 0 then
    (Algebra.ofId K F).comp <|
      (Algebra.botEquiv K L : (⊥ : Subalgebra K L) →ₐ[K] K).comp <| by
        /-
          F : Type u
          K : Type v
          L : Type w
          inst✝⁵ : Field K
          inst✝⁴ : Field L
          inst✝³ : Field F
          inst✝² : Algebra K L
          inst✝¹ : Algebra K F
          f : Polynomial K
          inst✝ : Polynomial.IsSplittingField K L f
          hf : Polynomial.Splits (algebraMap K F) f
          hf0 : Eq f 0
          ⊢ AlgHom K L (Subtype fun x => Membership.mem Bot.bot x)
        -/
        rw [← (splits_iff L f).1 (show f.Splits (RingHom.id K) from hf0.symm ▸ splits_zero _)]
        /-
          F : Type u
          K : Type v
          L : Type w
          inst✝⁵ : Field K
          inst✝⁴ : Field L
          inst✝³ : Field F
          inst✝² : Algebra K L
          inst✝¹ : Algebra K F
          f : Polynomial K
          inst✝ : Polynomial.IsSplittingField K L f
          hf : Polynomial.Splits (algebraMap K F) f
          hf0 : Eq f 0
          ⊢ AlgHom K L (Subtype fun x => Membership.mem Top.top x)
        -/
        exact Algebra.toTop
        /-
          🎉 no goals
        -/
  else AlgHom.comp (by
    /-
      F : Type u
      K : Type v
      L : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Field F
      inst✝² : Algebra K L
      inst✝¹ : Algebra K F
      f : Polynomial K
      inst✝ : Polynomial.IsSplittingField K L f
      hf : Polynomial.Splits (algebraMap K F) f
      hf0 : Not (Eq f 0)
      ⊢ AlgHom K (Subtype fun x => Membership.mem Top.top x) F
    -/
    rw [← adjoin_rootSet L f]
    exact Classical.choice (lift_of_splits _ fun y hy =>
      have : aeval y f = 0 := (eval₂_eq_eval_map _).trans <|
        (mem_roots <| map_ne_zero hf0).1 (Multiset.mem_toFinset.mp hy)
    ⟨IsAlgebraic.isIntegral ⟨f, hf0, this⟩,
      splits_of_splits_of_dvd _ hf0 hf <| minpoly.dvd _ _ this⟩)) Algebra.toTop


theorem finiteDimensional (f : K[X]) [IsSplittingField K L f] : FiniteDimensional K L := by
  classical
  exact ⟨@Algebra.top_toSubmodule K L _ _ _ ▸
    adjoin_rootSet L f ▸ fg_adjoin_of_finite (Finset.finite_toSet _) fun y hy ↦
      if hf : f = 0 then by rw [hf, rootSet_zero] at hy; cases hy
      else IsAlgebraic.isIntegral ⟨f, hf, (mem_rootSet'.mp hy).2⟩⟩


theorem of_algEquiv [Algebra K F] (p : K[X]) (f : F ≃ₐ[K] L) [IsSplittingField K F p] :
    IsSplittingField K L p := by
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Field F
    inst✝² : Algebra K L
    inst✝¹ : Algebra K F
    p : Polynomial K
    f : AlgEquiv K F L
    inst✝ : Polynomial.IsSplittingField K F p
    ⊢ Polynomial.IsSplittingField K L p
  -/
  constructor
    /-
      case splits'
      F : Type u
      K : Type v
      L : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Field F
      inst✝² : Algebra K L
      inst✝¹ : Algebra K F
      p : Polynomial K
      f : AlgEquiv K F L
      inst✝ : Polynomial.IsSplittingField K F p
      ⊢ Polynomial.Splits (algebraMap K L) p
    -/
  · rw [← f.toAlgHom.comp_algebraMap]
    /-
      case splits'
      F : Type u
      K : Type v
      L : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Field F
      inst✝² : Algebra K L
      inst✝¹ : Algebra K F
      p : Polynomial K
      f : AlgEquiv K F L
      inst✝ : Polynomial.IsSplittingField K F p
      ⊢ Polynomial.Splits ((↑↑f).comp (algebraMap K F)) p
    -/
    exact splits_comp_of_splits _ _ (splits F p)
    /-
      🎉 no goals
    -/
  · rw [← (AlgHom.range_eq_top f.toAlgHom).mpr f.surjective,
      adjoin_rootSet_eq_range (splits F p), adjoin_rootSet F p]


theorem adjoin_rootSet_eq_range [Algebra K F] (f : K[X]) [IsSplittingField K L f] (i : L →ₐ[K] F) :
    Algebra.adjoin K (rootSet f F) = i.range :=
  (Polynomial.adjoin_rootSet_eq_range (splits L f) i).mpr (adjoin_rootSet L f)


theorem IntermediateField.splits_of_splits (h : p.Splits (algebraMap K L))
    (hF : ∀ x ∈ p.rootSet L, x ∈ F) : p.Splits (algebraMap K F) := by
  classical
  simp_rw [← F.fieldRange_val, rootSet_def, Finset.mem_coe, Multiset.mem_toFinset] at hF
  exact splits_of_comp _ F.val.toRingHom h hF


theorem IntermediateField.splits_iff_mem (h : p.Splits (algebraMap K L)) :
    p.Splits (algebraMap K F) ↔ ∀ x ∈ p.rootSet L, x ∈ F := by
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    F : IntermediateField K L
    h : Polynomial.Splits (algebraMap K L) p
    ⊢ Iff (Polynomial.Splits (algebraMap K (Subtype fun x => Membership.mem F x))  …
  -/
  refine ⟨?_, IntermediateField.splits_of_splits h⟩
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    F : IntermediateField K L
    h : Polynomial.Splits (algebraMap K L) p
    ⊢ Polynomial.Splits (algebraMap K (Subtype fun x => Membership.mem F x)) p → ∀ …
  -/
  intro hF
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    F : IntermediateField K L
    h : Polynomial.Splits (algebraMap K L) p
    hF : Polynomial.Splits (algebraMap K (Subtype fun x => Membership.mem F x)) p
    ⊢ ∀ (x : L), Membership.mem (p.rootSet L) x → Membership.mem F x
  -/
  rw [← Polynomial.image_rootSet hF F.val, Set.forall_mem_image]
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    F : IntermediateField K L
    h : Polynomial.Splits (algebraMap K L) p
    hF : Polynomial.Splits (algebraMap K (Subtype fun x => Membership.mem F x)) p
    ⊢ ∀ ⦃x : Subtype fun x => Membership.mem F x⦄, Membership.mem (p.rootSet (Subt …
  -/
  exact fun x _ ↦ x.2
  /-
    🎉 no goals
  -/


theorem IsIntegral.mem_intermediateField_of_minpoly_splits {x : L} (int : IsIntegral K x)
    {F : IntermediateField K L} (h : Splits (algebraMap K F) (minpoly K x)) : x ∈ F := by
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    int : IsIntegral K x
    F : IntermediateField K L
    h : Polynomial.Splits (algebraMap K (Subtype fun x => Membership.mem F x)) (mi …
    ⊢ Membership.mem F x
  -/
  rw [← F.fieldRange_val]; exact int.mem_range_algebraMap_of_minpoly_splits h
                           /-
                             🎉 no goals
                           -/

