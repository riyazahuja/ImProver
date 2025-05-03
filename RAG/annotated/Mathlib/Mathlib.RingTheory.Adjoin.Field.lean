open AdjoinRoot in
/-- If `p` is the minimal polynomial of `a` over `F` then `F[a] ≃ₐ[F] F[x]/(p)` -/
def AlgEquiv.adjoinSingletonEquivAdjoinRootMinpoly {R : Type*} [CommRing R] [Algebra F R] (x : R) :
    Algebra.adjoin F ({x} : Set R) ≃ₐ[F] AdjoinRoot (minpoly F x) :=
  AlgEquiv.symm <| AlgEquiv.ofBijective (Minpoly.toAdjoin F x) <| by
    /-
      F : Type u_1
      inst✝² : Field F
      R : Type u_2
      inst✝¹ : CommRing R
      inst✝ : Algebra F R
      x : R
      ⊢ Function.Bijective ⇑(AdjoinRoot.Minpoly.toAdjoin F x)
    -/
    refine ⟨(injective_iff_map_eq_zero _).2 fun P₁ hP₁ ↦ ?_, Minpoly.toAdjoin.surjective F x⟩
    /-
      F : Type u_1
      inst✝² : Field F
      R : Type u_2
      inst✝¹ : CommRing R
      inst✝ : Algebra F R
      x : R
      P₁ : AdjoinRoot (minpoly F x)
      hP₁ : Eq ((AdjoinRoot.Minpoly.toAdjoin F x) P₁) 0
      ⊢ Eq P₁ 0
    -/
    obtain ⟨P, rfl⟩ := mk_surjective P₁
    /-
      case intro
      F : Type u_1
      inst✝² : Field F
      R : Type u_2
      inst✝¹ : CommRing R
      inst✝ : Algebra F R
      x : R
      P : Polynomial F
      hP₁ : Eq ((AdjoinRoot.Minpoly.toAdjoin F x) ((AdjoinRoot.mk (minpoly F x)) P)) 0
      ⊢ Eq ((AdjoinRoot.mk (minpoly F x)) P) 0
    -/
    refine AdjoinRoot.mk_eq_zero.mpr (minpoly.dvd F x ?_)
    /-
      case intro
      F : Type u_1
      inst✝² : Field F
      R : Type u_2
      inst✝¹ : CommRing R
      inst✝ : Algebra F R
      x : R
      P : Polynomial F
      hP₁ : Eq ((AdjoinRoot.Minpoly.toAdjoin F x) ((AdjoinRoot.mk (minpoly F x)) P)) 0
      ⊢ Eq ((Polynomial.aeval x) P) 0
    -/
    rwa [Minpoly.toAdjoin_apply', liftHom_mk, ← Subalgebra.coe_eq_zero, aeval_subalgebra_coe] at hP₁
    /-
      🎉 no goals
    -/


/-- Produce an algebra homomorphism `Adjoin R {x} →ₐ[R] T` sending `x` to
a root of `x`'s minimal polynomial in `T`. -/
noncomputable def Algebra.adjoin.liftSingleton {S T : Type*}
    [CommRing S] [CommRing T] [Algebra F S] [Algebra F T]
    (x : S) (y : T) (h : aeval y (minpoly F x) = 0) :
    Algebra.adjoin F {x} →ₐ[F] T :=
  (AdjoinRoot.liftHom _ y h).comp (AlgEquiv.adjoinSingletonEquivAdjoinRootMinpoly F x).toAlgHom


/-- If `K` and `L` are field extensions of `F` and we have `s : Finset K` such that
the minimal polynomial of each `x ∈ s` splits in `L` then `Algebra.adjoin F s` embeds in `L`. -/
theorem Polynomial.lift_of_splits {F K L : Type*} [Field F] [Field K] [Field L] [Algebra F K]
    [Algebra F L] (s : Finset K) : (∀ x ∈ s, IsIntegral F x ∧
      Splits (algebraMap F L) (minpoly F x)) → Nonempty (Algebra.adjoin F (s : Set K) →ₐ[F] L) := by
  classical
    refine Finset.induction_on s (fun _ ↦ ?_) fun a s _ ih H ↦ ?_
    · rw [coe_empty, Algebra.adjoin_empty]
      exact ⟨(Algebra.ofId F L).comp (Algebra.botEquiv F K)⟩
    rw [forall_mem_insert] at H
    rcases H with ⟨⟨H1, H2⟩, H3⟩
    cases' ih H3 with f
    choose H3 _ using H3
    rw [coe_insert, Set.insert_eq, Set.union_comm, Algebra.adjoin_union_eq_adjoin_adjoin]
    set Ks := Algebra.adjoin F (s : Set K)
    haveI : FiniteDimensional F Ks := ((Submodule.fg_iff_finiteDimensional _).1
      (fg_adjoin_of_finite s.finite_toSet H3)).of_subalgebra_toSubmodule
    letI := fieldOfFiniteDimensional F Ks
    letI := (f : Ks →+* L).toAlgebra
    have H5 : IsIntegral Ks a := H1.tower_top
    have H6 : (minpoly Ks a).Splits (algebraMap Ks L) := by
      refine splits_of_splits_of_dvd _ ((minpoly.monic H1).map (algebraMap F Ks)).ne_zero
        ((splits_map_iff _ _).2 ?_) (minpoly.dvd _ _ ?_)
      · rw [← IsScalarTower.algebraMap_eq]
        exact H2
      · rw [Polynomial.aeval_map_algebraMap, minpoly.aeval]
    obtain ⟨y, hy⟩ := Polynomial.exists_root_of_splits _ H6 (minpoly.degree_pos H5).ne'
    exact ⟨Subalgebra.ofRestrictScalars F _ <| Algebra.adjoin.liftSingleton Ks a y hy⟩


theorem IsIntegral.mem_range_algHom_of_minpoly_splits
    (int : IsIntegral R x) (h : Splits (algebraMap R K) (minpoly R x))(f : K →ₐ[R] L) :
    x ∈ f.range :=
  show x ∈ Set.range f from Set.image_subset_range _ ((minpoly R x).rootSet K) <| by
    /-
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra R K
      x : L
      inst✝ : Algebra R L
      int : IsIntegral R x
      h : Polynomial.Splits (algebraMap R K) (minpoly R x)
      f : AlgHom R K L
      ⊢ Membership.mem (Set.image (⇑f) ((minpoly R x).rootSet K)) x
    -/
    rw [image_rootSet h f, mem_rootSet']
    /-
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra R K
      x : L
      inst✝ : Algebra R L
      int : IsIntegral R x
      h : Polynomial.Splits (algebraMap R K) (minpoly R x)
      f : AlgHom R K L
      ⊢ And (Ne (Polynomial.map (algebraMap R L) (minpoly R x)) 0) (Eq ((Polynomial. …
    -/
    exact ⟨((minpoly.monic int).map _).ne_zero, minpoly.aeval R x⟩
    /-
      🎉 no goals
    -/


theorem IsIntegral.mem_range_algebraMap_of_minpoly_splits [Algebra K L] [IsScalarTower R K L]
    (int : IsIntegral R x) (h : Splits (algebraMap R K) (minpoly R x)) :
    x ∈ (algebraMap K L).range :=
  int.mem_range_algHom_of_minpoly_splits h (IsScalarTower.toAlgHom R K L)


theorem minpoly_neg_splits [Algebra K L] {x : L} (g : (minpoly K x).Splits (algebraMap K L)) :
    (minpoly K (-x)).Splits (algebraMap K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K (Neg.neg x))
  -/
  rw [minpoly.neg]
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (HMul.hMul (HPow.hPow (-1) (minpoly K x). …
  -/
  apply splits_mul _ _ g.comp_neg_X
  simpa only [map_pow, map_neg, map_one] using
    splits_C (algebraMap K L) ((-1) ^ (minpoly K x).natDegree)


theorem minpoly_add_algebraMap_splits [Algebra K L] {x : L} (r : K)
    (g : (minpoly K x).Splits (algebraMap K L)) :
    (minpoly K (x + algebraMap K L r)).Splits (algebraMap K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    r : K
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K (HAdd.hAdd x ((algebraMap K L) …
  -/
  simpa [minpoly.add_algebraMap] using g.comp_X_sub_C r
  /-
    🎉 no goals
  -/


theorem minpoly_sub_algebraMap_splits [Algebra K L] {x : L} (r : K)
    (g : (minpoly K x).Splits (algebraMap K L)) :
    (minpoly K (x - algebraMap K L r)).Splits (algebraMap K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    r : K
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K (HSub.hSub x ((algebraMap K L) …
  -/
  simpa only [sub_eq_add_neg, map_neg] using minpoly_add_algebraMap_splits (-r) g
  /-
    🎉 no goals
  -/


theorem minpoly_algebraMap_add_splits [Algebra K L] {x : L} (r : K)
    (g : (minpoly K x).Splits (algebraMap K L)) :
    (minpoly K (algebraMap K L r + x)).Splits (algebraMap K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    r : K
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K (HAdd.hAdd ((algebraMap K L) r …
  -/
  simpa only [add_comm] using minpoly_add_algebraMap_splits r g
  /-
    🎉 no goals
  -/


theorem minpoly_algebraMap_sub_splits [Algebra K L] {x : L} (r : K)
    (g : (minpoly K x).Splits (algebraMap K L)) :
    (minpoly K (algebraMap K L r - x)).Splits (algebraMap K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    r : K
    g : Polynomial.Splits (algebraMap K L) (minpoly K x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K (HSub.hSub ((algebraMap K L) r …
  -/
  simpa only [neg_sub] using minpoly_neg_splits (minpoly_sub_algebraMap_splits r g)
  /-
    🎉 no goals
  -/


/-- The `RingHom` version of `IsIntegral.minpoly_splits_tower_top`.  -/
theorem IsIntegral.minpoly_splits_tower_top' (int : IsIntegral R x) {f : K →+* L}
    (h : Splits (f.comp <| algebraMap R K) (minpoly R x)) :
    Splits f (minpoly K x) :=
  splits_of_splits_of_dvd _ ((minpoly.monic int).map _).ne_zero
    ((splits_map_iff _ _).mpr h) (minpoly.dvd_map_of_isScalarTower R _ x)


theorem IsIntegral.minpoly_splits_tower_top [Algebra K L] [Algebra R L] [IsScalarTower R K L]
    (int : IsIntegral R x) (h : Splits (algebraMap R L) (minpoly R x)) :
    Splits (algebraMap K L) (minpoly K x) := by
  /-
    R : Type u_1
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : CommRing M
    inst✝⁶ : Algebra R K
    inst✝⁵ : Algebra R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsScalarTower R K M
    x : M
    inst✝² : Algebra K L
    inst✝¹ : Algebra R L
    inst✝ : IsScalarTower R K L
    int : IsIntegral R x
    h : Polynomial.Splits (algebraMap R L) (minpoly R x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K x)
  -/
  rw [IsScalarTower.algebraMap_eq R K L] at h
  /-
    R : Type u_1
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : CommRing M
    inst✝⁶ : Algebra R K
    inst✝⁵ : Algebra R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsScalarTower R K M
    x : M
    inst✝² : Algebra K L
    inst✝¹ : Algebra R L
    inst✝ : IsScalarTower R K L
    int : IsIntegral R x
    h : Polynomial.Splits ((algebraMap K L).comp (algebraMap R K)) (minpoly R x)
    ⊢ Polynomial.Splits (algebraMap K L) (minpoly K x)
  -/
  exact int.minpoly_splits_tower_top' h
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a ring extension tower, `L` is a subalgebra of `K / F`,
then `[E[L] : E] ≤ [L : F]`. -/
lemma Subalgebra.adjoin_rank_le {F : Type*} (E : Type*) {K : Type*}
    [CommRing F] [StrongRankCondition F] [CommRing E] [StrongRankCondition E] [Ring K]
    [SMul F E] [Algebra E K] [Algebra F K] [IsScalarTower F E K]
    (L : Subalgebra F K) [Module.Free F L] :
    Module.rank E (Algebra.adjoin E (L : Set K)) ≤ Module.rank F L := by
  rw [← rank_toSubmodule, Module.Free.rank_eq_card_chooseBasisIndex F L,
    L.adjoin_eq_span_basis E (Module.Free.chooseBasis F L)]
  /-
    F : Type u_5
    E : Type u_6
    K : Type u_7
    inst✝⁹ : CommRing F
    inst✝⁸ : StrongRankCondition F
    inst✝⁷ : CommRing E
    inst✝⁶ : StrongRankCondition E
    inst✝⁵ : Ring K
    inst✝⁴ : SMul F E
    inst✝³ : Algebra E K
    inst✝² : Algebra F K
    inst✝¹ : IsScalarTower F E K
    L : Subalgebra F K
    inst✝ : Module.Free F (Subtype fun x => Membership.mem L x)
    ⊢ LE.le (Module.rank E (Subtype fun x => Membership.mem (Submodule.span E (Set …
  -/
  exact rank_span_le _ |>.trans Cardinal.mk_range_le
  /-
    🎉 no goals
  -/

