@[to_additive]
theorem mapClusterPt_atTop_zpow_iff_pow [DivInvMonoid G] [TopologicalSpace G] {x y : G} :
    MapClusterPt x atTop (y ^ · : ℤ → G) ↔ MapClusterPt x atTop (y ^ · : ℕ → G) := by
  /-
    G : Type u_1
    inst✝¹ : DivInvMonoid G
    inst✝ : TopologicalSpace G
    x y : G
    ⊢ Iff (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) (MapClusterPt x Fil …
  -/
  simp_rw [MapClusterPt, ← Nat.map_cast_int_atTop, map_map, comp_def, zpow_natCast]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mapClusterPt_self_zpow_atTop_pow (x : G) (m : ℤ) :
    MapClusterPt (x ^ m) atTop (x ^ · : ℕ → G) := by
  obtain ⟨y, hy⟩ : ∃ y, MapClusterPt y atTop (x ^ · : ℤ → G) :=
    exists_clusterPt_of_compactSpace _
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    m : Int
    y : G
    hy : MapClusterPt y Filter.atTop fun x_1 => HPow.hPow x x_1
    ⊢ MapClusterPt (HPow.hPow x m) Filter.atTop fun x_1 => HPow.hPow x x_1
  -/
  rw [← mapClusterPt_atTop_zpow_iff_pow]
  have H : MapClusterPt (x ^ m) (atTop.curry atTop) ↿(fun a b ↦ x ^ (m + b - a)) := by
    have : ContinuousAt (fun yz ↦ x ^ m * yz.2 / yz.1) (y, y) := by fun_prop
    simpa only [comp_def, ← zpow_sub, ← zpow_add, div_eq_mul_inv, Prod.map, mul_inv_cancel_right]
      using (hy.curry_prodMap hy).continuousAt_comp this
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    m : Int
    y : G
    hy : MapClusterPt y Filter.atTop fun x_1 => HPow.hPow x x_1
    H : MapClusterPt (HPow.hPow x m) (Filter.atTop.curry Filter.atTop) (Function.H …
    ⊢ MapClusterPt (HPow.hPow x m) Filter.atTop fun x_1 => HPow.hPow x x_1
  -/
  suffices Tendsto ↿(fun a b ↦ m + b - a) (atTop.curry atTop) atTop from H.of_comp this
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    m : Int
    y : G
    hy : MapClusterPt y Filter.atTop fun x_1 => HPow.hPow x x_1
    H : MapClusterPt (HPow.hPow x m) (Filter.atTop.curry Filter.atTop) (Function.H …
    ⊢ Filter.Tendsto (Function.HasUncurry.uncurry fun a b => HSub.hSub (HAdd.hAdd  …
  -/
  refine Tendsto.curry <| .of_forall fun a ↦ ?_
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    m : Int
    y : G
    hy : MapClusterPt y Filter.atTop fun x_1 => HPow.hPow x x_1
    H : MapClusterPt (HPow.hPow x m) (Filter.atTop.curry Filter.atTop) (Function.H …
    a : Int
    ⊢ Filter.Tendsto (fun b => HSub.hSub (HAdd.hAdd m b) a) Filter.atTop Filter.at …
  -/
  simp only [sub_eq_add_neg] -- TODO: add `Tendsto.atTop_sub_const` etc
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    m : Int
    y : G
    hy : MapClusterPt y Filter.atTop fun x_1 => HPow.hPow x x_1
    H : MapClusterPt (HPow.hPow x m) (Filter.atTop.curry Filter.atTop) (Function.H …
    a : Int
    ⊢ Filter.Tendsto (fun b => HAdd.hAdd (HAdd.hAdd m b) (Neg.neg a)) Filter.atTop …
  -/
  exact tendsto_atTop_add_const_right _ _ (tendsto_atTop_add_const_left atTop m tendsto_id)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mapClusterPt_one_atTop_pow (x : G) : MapClusterPt 1 atTop (x ^ · : ℕ → G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    ⊢ MapClusterPt 1 Filter.atTop fun x_1 => HPow.hPow x x_1
  -/
  simpa using mapClusterPt_self_zpow_atTop_pow x 0
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mapClusterPt_self_atTop_pow (x : G) : MapClusterPt x atTop (x ^ · : ℕ → G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    ⊢ MapClusterPt x Filter.atTop fun x_1 => HPow.hPow x x_1
  -/
  simpa using mapClusterPt_self_zpow_atTop_pow x 1
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mapClusterPt_atTop_pow_tfae (x y : G) :
    List.TFAE [
      MapClusterPt x atTop (y ^ · : ℕ → G),
      MapClusterPt x atTop (y ^ · : ℤ → G),
      x ∈ closure (range (y ^ · : ℕ → G)),
      x ∈ closure (range (y ^ · : ℤ → G)),
    ] := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x y : G
    ⊢ (List.cons (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) (List.cons ( …
  -/
  tfae_have 2 ↔ 1 := mapClusterPt_atTop_zpow_iff_pow
  tfae_have 3 → 4 := by
    refine fun h ↦ closure_mono (range_subset_iff.2 fun n ↦ ?_) h
    exact ⟨n, zpow_natCast _ _⟩
  tfae_have 4 → 1 := by
    refine fun h ↦ closure_minimal ?_ isClosed_setOf_clusterPt h
    exact range_subset_iff.2 (mapClusterPt_self_zpow_atTop_pow _)
  tfae_have 1 → 3 := by
    rw [mem_closure_iff_clusterPt]
    exact (ClusterPt.mono · (le_principal_iff.2 range_mem_map))
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x y : G
    tfae_2_iff_1 : Iff (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) (MapCl …
    tfae_3_to_4 : Membership.mem (closure (Set.range fun x => HPow.hPow y x)) x →  …
    tfae_4_to_1 : Membership.mem (closure (Set.range fun x => HPow.hPow y x)) x →  …
    tfae_1_to_3 : (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) → Membershi …
    ⊢ (List.cons (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) (List.cons ( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mapClusterPt_atTop_pow_iff_mem_topologicalClosure_zpowers {x y : G} :
    MapClusterPt x atTop (y ^ · : ℕ → G) ↔ x ∈ (Subgroup.zpowers y).topologicalClosure :=
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x y : G
    ⊢ Eq ((List.cons (MapClusterPt x Filter.atTop fun x => HPow.hPow y x) (List.co …
  -/
  /-
    🎉 no goals
  -/
  (mapClusterPt_atTop_pow_tfae x y).out 0 3
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mapClusterPt_inv_atTop_pow {x y : G} :
    MapClusterPt x⁻¹ atTop (y ^ · : ℕ → G) ↔ MapClusterPt x atTop (y ^ · : ℕ → G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x y : G
    ⊢ Iff (MapClusterPt (Inv.inv x) Filter.atTop fun x => HPow.hPow y x) (MapClust …
  -/
  simp only [mapClusterPt_atTop_pow_iff_mem_topologicalClosure_zpowers, inv_mem_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_range_zpow_eq_pow (x : G) :
    closure (range (x ^ · : ℤ → G)) = closure (range (x ^ · : ℕ → G)) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    ⊢ Eq (closure (Set.range fun x_1 => HPow.hPow x x_1)) (closure (Set.range fun  …
  -/
  ext y
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x y : G
    ⊢ Iff (Membership.mem (closure (Set.range fun x_1 => HPow.hPow x x_1)) y) (Mem …
  -/
  exact (mapClusterPt_atTop_pow_tfae y x).out 3 2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem denseRange_zpow_iff_pow {x : G} :
    DenseRange (x ^ · : ℤ → G) ↔ DenseRange (x ^ · : ℕ → G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    x : G
    ⊢ Iff (DenseRange fun x_1 => HPow.hPow x x_1) (DenseRange fun x_1 => HPow.hPow …
  -/
  simp only [DenseRange, dense_iff_closure_eq, closure_range_zpow_eq_pow]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem topologicalClosure_subgroupClosure_toSubmonoid (s : Set G) :
    (Subgroup.closure s).toSubmonoid.topologicalClosure =
      (Submonoid.closure s).topologicalClosure := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    ⊢ Eq (Subgroup.closure s).topologicalClosure (Submonoid.closure s).topological …
  -/
  refine le_antisymm ?_ (closure_mono <| Subgroup.le_closure_toSubmonoid _)
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    ⊢ LE.le (Subgroup.closure s).topologicalClosure (Submonoid.closure s).topologi …
  -/
  refine Submonoid.topologicalClosure_minimal _ ?_ isClosed_closure
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    ⊢ LE.le (Subgroup.closure s).toSubmonoid (Submonoid.closure s).topologicalClos …
  -/
  rw [Subgroup.closure_toSubmonoid, Submonoid.closure_le]
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    ⊢ HasSubset.Subset (Union.union s (Inv.inv s)) ↑(Submonoid.closure s).topologi …
  -/
  refine union_subset (Submonoid.subset_closure.trans subset_closure) fun x hx ↦ ?_
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    x : G
    hx : Membership.mem (Inv.inv s) x
    ⊢ Membership.mem (↑(Submonoid.closure s).topologicalClosure) x
  -/
  refine closure_mono (Submonoid.powers_le.2 (Submonoid.subset_closure <| Set.mem_inv.1 hx)) ?_
  rw [Submonoid.coe_powers, ← closure_range_zpow_eq_pow, ← Subgroup.coe_zpowers,
    ← Subgroup.topologicalClosure_coe, SetLike.mem_coe, ← inv_mem_iff]
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    x : G
    hx : Membership.mem (Inv.inv s) x
    ⊢ Membership.mem (Subgroup.zpowers (Inv.inv x)).topologicalClosure (Inv.inv x)
  -/
  exact subset_closure <| Subgroup.mem_zpowers _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_submonoidClosure_eq_closure_subgroupClosure (s : Set G) :
    closure (Submonoid.closure s : Set G) = closure (Subgroup.closure s) :=
  congrArg SetLike.coe (topologicalClosure_subgroupClosure_toSubmonoid s).symm


@[to_additive]
theorem dense_submonoidClosure_iff_subgroupClosure {s : Set G} :
    Dense (Submonoid.closure s : Set G) ↔ Dense (Subgroup.closure s : Set G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : CompactSpace G
    inst✝ : TopologicalGroup G
    s : Set G
    ⊢ Iff (Dense ↑(Submonoid.closure s)) (Dense ↑(Subgroup.closure s))
  -/
  simp only [dense_iff_closure_eq, closure_submonoidClosure_eq_closure_subgroupClosure]
  /-
    🎉 no goals
  -/

